// (c) Copyright 2025 Mikołaj Kuranowski
// SPDX-License-Identifier: MIT

use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::Path;
use std::sync::Arc;

use graph_builder::GraphBuilder;

use crate::osm::Profile;
use crate::Graph;

mod graph_builder;
pub mod model;
pub mod pbf;
pub mod xml;

/// Error which can occur during OSM reading and parsing.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] Arc<io::Error>),

    #[error("xml: {0}")]
    Xml(quick_xml::Error),

    #[error("pbf: {0}")]
    Pbf(pbf::Error),

    #[error("unknown file format: data does not look like .osm/.osm.gz/osm.bz2/.osm.pbf")]
    UnknownFileFormat,
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(Arc::new(e))
    }
}

impl From<quick_xml::Error> for Error {
    fn from(e: quick_xml::Error) -> Self {
        match e {
            quick_xml::Error::Io(ioe) => Error::Io(ioe),
            _ => Error::Xml(e),
        }
    }
}

impl From<pbf::Error> for Error {
    fn from(e: pbf::Error) -> Self {
        match e {
            pbf::Error::Io(ioe) => Error::Io(ioe),
            _ => Error::Pbf(e),
        }
    }
}

/// Format of the input OSM file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// Unknown format - guess the format based on the content
    Unknown,

    /// Force uncompressed [OSM XML](https://wiki.openstreetmap.org/wiki/OSM_XML)
    Xml,

    /// Force [OSM XML](https://wiki.openstreetmap.org/wiki/OSM_XML)
    /// with [gzip](https://en.wikipedia.org/wiki/Gzip) compression
    XmlGz,

    /// Force [OSM XML](https://wiki.openstreetmap.org/wiki/OSM_XML)
    /// with [bzip2](https://en.wikipedia.org/wiki/Bzip2) compression
    XmlBz2,

    /// Force [OSM PBF](https://wiki.openstreetmap.org/wiki/PBF_Format)
    Pbf,
}

impl FileFormat {
    /// Attempts to detect the file format based on the initial bytes of the file.
    /// At least 8 bytes should be provided.
    pub fn detect(b: &[u8]) -> FileFormat {
        if b.starts_with(b"<?xml") || b.starts_with(b"<osm") {
            FileFormat::Xml
        } else if b.starts_with(b"\x1F\x8B") {
            FileFormat::XmlGz // Gzip magic bytes
        } else if b.starts_with(b"BZh") {
            FileFormat::XmlBz2 // Bzip2 magic bytes
        } else if b.len() >= 8 && &b[4..8] == b"\x0A\x09OS" {
            // OSM PBF always starts with the first 4 bytes encoding the BlobHeader length - we ignore this,
            // rather, we check the first field of the first BlobHeader, which should be:
            // field 1, type string, "OSMHeader" (length 9). - ? ? ? ? 0x0A 0x09 O S M H e a d e r
            FileFormat::Pbf
        } else {
            FileFormat::Unknown
        }
    }
}

/// Additional controls for interpreting OSM data as a routing [Graph].
#[derive(Debug)]
pub struct Options<'a> {
    /// How OSM features should be interpreted and converted into a [Graph].
    pub profile: &'a Profile<'a>,

    /// Format of the input data. Currently, only [FileFormat::Xml] is supported.
    pub file_format: FileFormat,

    /// Filter features by a specific bounding box. In order: left (min lon), bottom (min lat),
    /// right (max lon), top (max lat). Ignored if all values are set to zero.
    pub bbox: [f32; 4],

    /// Tags to save from nodes
    pub node_tag_filters: &'a [super::NodeTagFilter],
}

/// Helper function to open a file and run a callback over its features.
/// This allows us to re-open the file multiple times for Two-Pass architecture.
fn with_file_iterator<P: AsRef<Path>, F, T>(path: P, format: FileFormat, mut f: F) -> Result<T, Error>
where
    F: FnMut(&mut dyn Iterator<Item = Result<model::Feature, Error>>) -> Result<T, Error>,
{
    let mut file = File::open(path.as_ref())?;
    
    let detected_format = if format == FileFormat::Unknown {
        let mut header = [0u8; 8];
        let bytes_read = file.read(&mut header)?;
        file.seek(io::SeekFrom::Start(0))?;
        FileFormat::detect(&header[..bytes_read])
    } else {
        format
    };

    let b = io::BufReader::new(file);

    match detected_format {
        FileFormat::Unknown => Err(Error::UnknownFileFormat),
        FileFormat::Xml => {
            let mut iter = xml::features_from_file(b).map(|r| r.map_err(Error::from));
            f(&mut iter)
        }
        FileFormat::XmlGz => {
            let d = flate2::bufread::MultiGzDecoder::new(b);
            let mut iter = xml::features_from_file(io::BufReader::new(d)).map(|r| r.map_err(Error::from));
            f(&mut iter)
        }
        FileFormat::XmlBz2 => {
            let d = bzip2::bufread::MultiBzDecoder::new(b);
            let mut iter = xml::features_from_file(io::BufReader::new(d)).map(|r| r.map_err(Error::from));
            f(&mut iter)
        }
        FileFormat::Pbf => {
            let mut iter = pbf::features_from_file(b).map(|r| r.map_err(Error::from));
            f(&mut iter)
        }
    }
}

/// Helper function to read from a buffer and run a callback over its features.
fn with_buffer_iterator<F, T>(data: &[u8], format: FileFormat, mut f: F) -> Result<T, Error>
where
    F: FnMut(&mut dyn Iterator<Item = Result<model::Feature, Error>>) -> Result<T, Error>,
{
    let detected_format = if format == FileFormat::Unknown {
        FileFormat::detect(data)
    } else {
        format
    };

    match detected_format {
        FileFormat::Unknown => Err(Error::UnknownFileFormat),
        FileFormat::Xml => {
            let mut iter = xml::features_from_buffer(data).map(|r| r.map_err(Error::from));
            f(&mut iter)
        }
        FileFormat::XmlGz => {
            let d = flate2::bufread::MultiGzDecoder::new(io::Cursor::new(data));
            let mut iter = xml::features_from_file(io::BufReader::new(d)).map(|r| r.map_err(Error::from));
            f(&mut iter)
        }
        FileFormat::XmlBz2 => {
            let d = bzip2::bufread::MultiBzDecoder::new(io::Cursor::new(data));
            let mut iter = xml::features_from_file(io::BufReader::new(d)).map(|r| r.map_err(Error::from));
            f(&mut iter)
        }
        FileFormat::Pbf => {
            let mut iter = pbf::features_from_file(io::Cursor::new(data)).map(|r| r.map_err(Error::from));
            f(&mut iter)
        }
    }
}

/// Parse OSM features from an IO stream into a [Graph] as per the provided [Options].
///
/// NOTE: Because the Two-Pass architecture requires traversing the data twice, this function 
/// buffers the entire stream into RAM first! For large files (e.g. >100MB), you should 
/// strictly use `add_features_from_file` instead.
pub fn add_features_from_io<'a, R: io::BufRead>(
    g: &'a mut Graph,
    options: &'a Options<'a>,
    mut reader: R,
) -> Result<(), Error> {
    log::warn!(target: "routx::osm", "add_features_from_io buffers the entire stream into RAM for Two-Pass processing. Use add_features_from_file for large maps.");
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    add_features_from_buffer(g, options, &data)
}

/// Parse OSM features from a file at the provided path into a [Graph] as per the provided [Options].
/// Uses a Two-Pass architecture to drastically reduce RAM usage.
pub fn add_features_from_file<'a, P: AsRef<Path>>(
    g: &'a mut Graph,
    options: &'a Options<'a>,
    path: P,
) -> Result<(), Error> {
    let mut needed_nodes = rustc_hash::FxHashSet::default();

    // Pass 1: Identify all nodes utilized by routable ways.
    with_file_iterator(path.as_ref(), options.file_format, |features| {
        graph_builder::pass1(options, features, &mut needed_nodes)
    })?;

    // Pass 2: Actually build the graph loading only needed nodes and ways.
    with_file_iterator(path.as_ref(), options.file_format, |features| {
        let mut builder = GraphBuilder::new(g, options, std::mem::take(&mut needed_nodes));
        builder.add_features(features)
    })?;

    Ok(())
}

/// Parse OSM features from a static buffer into a [Graph] as per the provided [Options].
/// Uses a Two-Pass architecture.
pub fn add_features_from_buffer<'a>(
    g: &'a mut Graph,
    options: &'a Options<'a>,
    data: &[u8],
) -> Result<(), Error> {
    let mut needed_nodes = rustc_hash::FxHashSet::default();

    // Pass 1: Identify all nodes utilized by routable ways.
    with_buffer_iterator(data, options.file_format, |features| {
        graph_builder::pass1(options, features, &mut needed_nodes)
    })?;

    // Pass 2: Actually build the graph loading only needed nodes and ways.
    with_buffer_iterator(data, options.file_format, |features| {
        let mut builder = GraphBuilder::new(g, options, std::mem::take(&mut needed_nodes));
        builder.add_features(features)
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_format_detect() {
        assert_eq!(FileFormat::detect(b""), FileFormat::Unknown);
        assert_eq!(FileFormat::detect(b"lorem ipsum dolo"), FileFormat::Unknown);
        assert_eq!(FileFormat::detect(b"<?xml version='1"), FileFormat::Xml);
        assert_eq!(FileFormat::detect(b"<osm version='0."), FileFormat::Xml);
        assert_eq!(
            FileFormat::detect(b"\x1F\x8B\x08\x08\x84s\xCE^"),
            FileFormat::XmlGz,
        );
        assert_eq!(
            FileFormat::detect(b"BZh91AY&SY\x12\x10&X\x00\x04"),
            FileFormat::XmlBz2,
        );
        assert_eq!(
            FileFormat::detect(b"\x00\x00\x00\x0D\x0A\x09OSMHeader\x18"),
            FileFormat::Pbf,
        );
    }
}
