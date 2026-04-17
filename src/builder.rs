use std::collections::HashMap;
use std::io::{self, Write};
use std::mem::size_of;

use crate::{Graph, KDTree};
use crate::flat_graph::{FlatEdge, FlatKDNode, FlatNode, FlatTag, Header, MAGIC, VERSION};

struct StringPool {
    buffer: Vec<u8>,
    map: HashMap<String, u32>,
}

impl StringPool {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            map: HashMap::new(),
        }
    }

    /// Inserts a string into the pool. Returns its starting offset (in bytes).
    fn insert(&mut self, s: &str) -> u32 {
        if let Some(&offset) = self.map.get(s) {
            offset
        } else {
            let offset = self.buffer.len() as u32;
            self.buffer.extend_from_slice(s.as_bytes());
            self.buffer.push(0); // Null-terminator for easy reading from C / FFI
            self.map.insert(s.to_string(), offset);
            offset
        }
    }
}

/// Transforms a traditional `Graph` from RAM to a flat zero-copy layout,
/// and then writes it directly to the provided Writer/File using io::Write.
pub fn build_memory_mapped_graph<W: Write>(
    graph: &Graph,
    kd_tree: Option<&KDTree>,
    mut writer: W,
) -> io::Result<()> {
    let mut string_pool = StringPool::new();

    let mut flat_nodes = Vec::with_capacity(graph.len());
    let mut flat_edges = Vec::new();
    let mut flat_tags = Vec::new();

    let mut node_id_to_idx = HashMap::with_capacity(graph.len());
    for (idx, &old_id) in graph.0.keys().enumerate() {
        node_id_to_idx.insert(old_id, idx as u32);
    }

    for (&old_id, (node, edges)) in &graph.0 {
        let first_edge_idx = flat_edges.len() as u32;
        let mut edge_count = 0;

        for edge in edges {
            // Ignore edges leading to removed nodes or nodes outside the bbox
            if let Some(&to_idx) = node_id_to_idx.get(&edge.to) {
                flat_edges.push(FlatEdge {
                    to_node_idx: to_idx,
                    cost: edge.cost,
                });
                edge_count += 1;
            }
        }

        let first_tag_idx = flat_tags.len() as u32;
        let mut tag_count = 0;

        if let Some(tags) = graph.1.get(&old_id) {
            for (k, v) in tags {
                let k_offset = string_pool.insert(k);
                let v_offset = string_pool.insert(v);
                flat_tags.push(FlatTag {
                    key_string_offset: k_offset,
                    val_string_offset: v_offset,
                });
                tag_count += 1;
            }
        }

        // Clamp counts to 5 bits maximum (31 elements) to support bit-packing format
        let clamped_edge_count = std::cmp::min(edge_count, 31);
        if edge_count > 31 {
            log::warn!(target: "routx::builder", "Node {} exceeds max packed edge capacity ({} > 31). Truncating.", old_id, edge_count);
        }
        
        let clamped_tag_count = std::cmp::min(tag_count, 31);
        if tag_count > 31 {
            log::warn!(target: "routx::builder", "Node {} exceeds max packed tag capacity ({} > 31). Truncating.", old_id, tag_count);
        }

        let edge_info = (first_edge_idx << 5) | clamped_edge_count;
        let tag_info = (first_tag_idx << 5) | clamped_tag_count;

        flat_nodes.push(FlatNode {
            osm_id: node.osm_id,
            lat: node.lat,
            lon: node.lon,
            edge_info,
            tag_info,
        });
    }

    let flat_kd_nodes = if let Some(tree) = kd_tree {
        tree.flatten(&node_id_to_idx)
    } else {
        Vec::new()
    };

    let header_size = size_of::<Header>() as u64;
    let nodes_size = (flat_nodes.len() * size_of::<FlatNode>()) as u64;
    let edges_size = (flat_edges.len() * size_of::<FlatEdge>()) as u64;
    let tags_size = (flat_tags.len() * size_of::<FlatTag>()) as u64;
    let kd_tree_size = (flat_kd_nodes.len() * size_of::<FlatKDNode>()) as u64;

    let nodes_offset = header_size;
    let edges_offset = nodes_offset + nodes_size;
    let tags_offset = edges_offset + edges_size;
    let kd_tree_offset = tags_offset + tags_size;
    let strings_offset = kd_tree_offset + kd_tree_size;

    let header = Header {
        magic: MAGIC,
        version: VERSION,
        nodes_offset,
        nodes_count: flat_nodes.len() as u32,
        _pad1: 0,
        edges_offset,
        edges_count: flat_edges.len() as u32,
        _pad2: 0,
        tags_offset,
        tags_count: flat_tags.len() as u32,
        _pad3: 0,
        kd_tree_offset,
        kd_tree_count: flat_kd_nodes.len() as u32,
        _pad4: 0,
        strings_offset,
        strings_length: string_pool.buffer.len() as u32,
        _pad5: 0,
    };

    fn write_struct<T, W: Write>(writer: &mut W, data: &T) -> io::Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data as *const T as *const u8,
                size_of::<T>(),
            )
        };
        writer.write_all(bytes)
    }

    fn write_slice<T, W: Write>(writer: &mut W, data: &[T]) -> io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * size_of::<T>(),
            )
        };
        writer.write_all(bytes)
    }

    write_struct(&mut writer, &header)?;
    write_slice(&mut writer, &flat_nodes)?;
    write_slice(&mut writer, &flat_edges)?;
    write_slice(&mut writer, &flat_tags)?;
    write_slice(&mut writer, &flat_kd_nodes)?;
    writer.write_all(&string_pool.buffer)?;

    Ok(())
}
