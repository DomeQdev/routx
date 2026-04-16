use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use routx::flat_graph::MemoryMappedGraph;
use routx::osm::NodeTagFilter;
use routx::{build_memory_mapped_graph, KDTree};

#[derive(Debug, thiserror::Error)]
#[error("{0}: {1}")]
struct GraphLoadError(PathBuf, #[source] routx::osm::Error);

#[derive(Parser)]
#[command(name = "routx", about = "Routing engine for OSM data")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Build {
        osm_file: PathBuf,
        out_file: PathBuf,
        #[arg(long)]
        tag_filter: Vec<String>,
    },

    Route {
        routx_file: PathBuf,

        start_lat: f32,
        start_lon: f32,
        end_lat: f32,
        end_lon: f32,
    },
}

fn parse_tag_filters(filters: &[String]) -> Vec<NodeTagFilter> {
    let mut parsed = Vec::new();
    for f in filters {
        // Format: key=value:tag1,tag2
        if let Some((kv, tags)) = f.split_once(':') {
            if let Some((k, v)) = kv.split_once('=') {
                let tags_to_save: Vec<String> = tags.split(',').map(|s| s.to_string()).collect();
                parsed.push(NodeTagFilter {
                    key: k.to_string(),
                    value: v.to_string(),
                    tags_to_save,
                });
            }
        }
    }
    parsed
}

pub fn main() -> Result<(), Box<dyn Error>> {
    colog::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Build {
            osm_file,
            out_file,
            tag_filter,
        } => {
            log::info!(target: "routx", "Ładowanie danych OSM...");
            let filters = parse_tag_filters(&tag_filter);
            let g = load_graph(&osm_file, &filters)?;

            log::info!(target: "routx", "Budowanie KD-Tree...");
            let kd_tree = KDTree::build_from_graph(&g);

            log::info!(target: "routx", "Spłaszczanie i zapisywanie do {:?}...", out_file);
            let file = File::create(out_file)?;
            let writer = BufWriter::new(file);

            build_memory_mapped_graph(&g, kd_tree.as_ref(), writer)?;
            log::info!(target: "routx", "Gotowe!");
        }

        Commands::Route {
            routx_file,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
        } => {
            log::info!(target: "routx", "Ładowanie zmapowanej mapy z {:?}...", routx_file);
            // W CLI wczytujemy plik w całości do wektora dla ułatwienia. 
            // W C/Node.js użyjesz natywnego mmap/SharedArrayBuffer bez alokacji.
            let data = std::fs::read(routx_file)?;
            let mmap = MemoryMappedGraph::new(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            // Znalezienie startowego i końcowego indeksu
            log::info!(target: "routx", "Szukanie najbliższych węzłów...");
            
            // Szukamy w szerokim promieniu (np. 10km) i wybieramy najbliższy
            let find_closest = |lat, lon| -> Result<u32, Box<dyn Error>> {
                let candidates = mmap.find_nodes_within_radius(lat, lon, 10000.0);
                candidates.into_iter()
                    .min_by(|&a, &b| {
                        let na = mmap.get_node(a).unwrap();
                        let nb = mmap.get_node(b).unwrap();
                        let dist_a = routx::earth_distance(lat, lon, na.lat, na.lon);
                        let dist_b = routx::earth_distance(lat, lon, nb.lat, nb.lon);
                        dist_a.partial_cmp(&dist_b).unwrap()
                    })
                    .ok_or_else(|| "Nie znaleziono węzłów w pobliżu 10km".into())
            };

            let start_idx = find_closest(start_lat, start_lon)?;
            let end_idx = find_closest(end_lat, end_lon)?;

            log::info!(target: "routx", "Uruchamianie algorytmu A*...");
            let route_indices = routx::find_route_without_turn_around(
                &mmap,
                start_idx,
                end_idx,
                routx::DEFAULT_STEP_LIMIT,
            )?;

            // Wypisanie wyniku
            println!("{{");
            println!("  \"type\": \"FeatureCollection\",");
            println!("  \"features\": [");
            println!("    {{");
            println!("      \"type\": \"Feature\",");
            println!("      \"properties\": {{}},");
            println!("      \"geometry\": {{");
            println!("        \"type\": \"LineString\",");
            println!("        \"coordinates\": [");

            let mut iter = route_indices.iter().peekable();
            while let Some(&idx) = iter.next() {
                let node = mmap.get_node(idx).unwrap();
                let suffix = if iter.peek().is_some() { "," } else { "" };
                println!("          [{}, {}]{}", node.lon, node.lat, suffix);
            }

            println!("        ]");
            println!("      }}");
            println!("    }}");
            println!("  ]");
            println!("}}");
        }
    }

    Ok(())
}

fn load_graph<P: AsRef<Path>>(
    path: P,
    filters: &[NodeTagFilter],
) -> Result<routx::Graph, GraphLoadError> {
    let mut g = routx::Graph::default();
    // Do odczytu źródłowego używamy formatu np. PBF, domyślny profil samochodu
    let options = routx::osm::Options {
        profile: &routx::osm::CAR_PROFILE,
        file_format: routx::osm::FileFormat::Unknown,
        bbox: [0.0; 4],
        node_tag_filters: filters,
    };
    match routx::osm::add_features_from_file(&mut g, &options, path.as_ref()) {
        Ok(()) => Ok(g),
        Err(e) => Err(GraphLoadError(PathBuf::from(path.as_ref()), e)),
    }
}
