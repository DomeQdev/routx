use std::mem::size_of;
use std::str;

pub const MAGIC: [u8; 6] = *b"ROUTX\0";
pub const VERSION: u16 = 1;

/// Index indicating the lack of an element (e.g. lack of left/right child in KDTree)
pub const NULL_IDX: u32 = u32::MAX;

/// Binary file header. Designed to eliminate hidden padding
/// (all u64 are 8-byte aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub magic: [u8; 6],
    pub version: u16,           // offset: 6

    pub nodes_offset: u64,      // offset: 8
    pub nodes_count: u32,       // offset: 16
    pub _pad1: u32,             // offset: 20

    pub edges_offset: u64,      // offset: 24
    pub edges_count: u32,       // offset: 32
    pub _pad2: u32,             // offset: 36

    pub tags_offset: u64,       // offset: 40
    pub tags_count: u32,        // offset: 48
    pub _pad3: u32,             // offset: 52

    pub kd_tree_offset: u64,    // offset: 56
    pub kd_tree_count: u32,     // offset: 64
    pub _pad4: u32,             // offset: 68

    pub strings_offset: u64,    // offset: 72
    pub strings_length: u32,    // offset: 80
    pub _pad5: u32,             // offset: 84
}                               // total size: 88 bytes

/// Flat representation of a node (Node). Size: 24 bytes (Bit-packed for compression).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlatNode {
    pub osm_id: i64,            // 8 bytes
    pub lat: f32,               // 4 bytes
    pub lon: f32,               // 4 bytes
    
    /// Bit-packed info: first 27 bits for first_edge_idx, last 5 bits for edge_count
    pub edge_info: u32,         // 4 bytes
    
    /// Bit-packed info: first 27 bits for first_tag_idx, last 5 bits for tag_count
    pub tag_info: u32,          // 4 bytes
}

impl FlatNode {
    #[inline]
    pub fn edge_count(&self) -> u32 {
        self.edge_info & 0x1F
    }

    #[inline]
    pub fn first_edge_idx(&self) -> u32 {
        self.edge_info >> 5
    }

    #[inline]
    pub fn tag_count(&self) -> u32 {
        self.tag_info & 0x1F
    }

    #[inline]
    pub fn first_tag_idx(&self) -> u32 {
        self.tag_info >> 5
    }
}

/// Flat representation of an edge (Edge). Size: 8 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlatEdge {
    pub to_node_idx: u32,
    pub cost: f32,
}

/// Pointers to strings storing a tag for a given Node. Size: 8 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlatTag {
    pub key_string_offset: u32,
    pub val_string_offset: u32,
}

/// Element of the flat KD tree (KD-Tree). Size: 12 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlatKDNode {
    pub node_idx: u32,
    pub left_idx: u32, // NULL_IDX (u32::MAX) if None
    pub right_idx: u32, // NULL_IDX (u32::MAX) if None
}

/// Wrapper for raw mapped memory. Provides zero-copy access to the graph.
pub struct MemoryMappedGraph<'a> {
    _data: &'a [u8],
    header: &'a Header,
    nodes: &'a [FlatNode],
    edges: &'a [FlatEdge],
    tags: &'a [FlatTag],
    kd_tree: &'a [FlatKDNode],
    strings: &'a str,
}

impl<'a> MemoryMappedGraph<'a> {
    /// Initializes the graph directly from a memory buffer (e.g. mmap or SharedArrayBuffer).
    pub fn new(data: &'a [u8]) -> Result<Self, &'static str> {
        if data.len() < size_of::<Header>() {
            return Err("Data buffer is too short for header");
        }

        // SAFETY: The length is checked above. Align is checked inherently if the underlying buffer is well-aligned,
        // but usually mmap returns page-aligned memory. 
        let header = unsafe { &*(data.as_ptr() as *const Header) };

        if header.magic != MAGIC {
            return Err("Invalid magic bytes (not a .routx file)");
        }
        if header.version != VERSION {
            return Err("Unsupported file version");
        }

        // Helper to safely extract slices
        unsafe fn get_slice<T>(data: &[u8], offset: u64, count: u32) -> Result<&[T], &'static str> {
            let offset = offset as usize;
            let count = count as usize;
            let byte_len = count * size_of::<T>();
            
            if offset + byte_len > data.len() {
                return Err("Array bounds exceeded memory limit");
            }
            if offset % std::mem::align_of::<T>() != 0 {
                return Err("Memory is misaligned");
            }
            
            Ok(std::slice::from_raw_parts(
                data.as_ptr().add(offset) as *const T,
                count,
            ))
        }

        let nodes = unsafe { get_slice::<FlatNode>(data, header.nodes_offset, header.nodes_count)? };
        let edges = unsafe { get_slice::<FlatEdge>(data, header.edges_offset, header.edges_count)? };
        let tags = unsafe { get_slice::<FlatTag>(data, header.tags_offset, header.tags_count)? };
        let kd_tree = unsafe { get_slice::<FlatKDNode>(data, header.kd_tree_offset, header.kd_tree_count)? };

        let strings_start = header.strings_offset as usize;
        let strings_end = strings_start + header.strings_length as usize;
        if strings_end > data.len() {
            return Err("Strings pool bounds exceeded");
        }
        
        let strings_slice = &data[strings_start..strings_end];
        let strings = str::from_utf8(strings_slice).map_err(|_| "Invalid UTF-8 in strings pool")?;

        Ok(Self {
            _data: data,
            header,
            nodes,
            edges,
            tags,
            kd_tree,
            strings,
        })
    }

    /// Retrieves a Node based on its index
    #[inline]
    pub fn get_node(&self, idx: u32) -> Option<&FlatNode> {
        self.nodes.get(idx as usize)
    }

    /// Retrieves Edges for a given Node
    #[inline]
    pub fn get_edges(&self, node: &FlatNode) -> &[FlatEdge] {
        let start = node.first_edge_idx() as usize;
        let end = start + node.edge_count() as usize;
        &self.edges[start..end]
    }

    /// Iterates over decoded Node Tags (Returns Strings: key, value)
    pub fn get_tags(&self, node: &FlatNode) -> impl Iterator<Item = (&'a str, &'a str)> + '_ {
        let start = node.first_tag_idx() as usize;
        let end = start + node.tag_count() as usize;
        self.tags[start..end].iter().map(move |tag| {
            let key = self.get_string(tag.key_string_offset);
            let val = self.get_string(tag.val_string_offset);
            (key, val)
        })
    }

    /// Reads a character string (String) from the pool based on the offset
    #[inline]
    pub fn get_string(&self, offset: u32) -> &'a str {
        let offset = offset as usize;
        if offset >= self.strings.len() {
            return "";
        }
        let slice = &self.strings[offset..];
        let end = slice.find('\0').unwrap_or(slice.len());
        &slice[..end]
    }

    /// Number of nodes in the graph
    #[inline]
    pub fn nodes_count(&self) -> u32 {
        self.header.nodes_count
    }

    /// Searches for all nodes within a given radius (radius in meters).
    /// Uses a flat KD-Tree for optimal $O(\log n)$ search.
    pub fn find_nodes_within_radius(&self, lat: f32, lon: f32, radius_meters: f32) -> Vec<u32> {
        let mut result = Vec::new();
        if self.kd_tree.is_empty() {
            return result;
        }

        // Stack prevents Call Stack overflow for deep trees
        let mut stack = vec![(0u32, false)]; // (kd_node_idx, lon_divides)

        while let Some((kd_idx, lon_divides)) = stack.pop() {
            if kd_idx == NULL_IDX {
                continue;
            }
            
            let kd_node = &self.kd_tree[kd_idx as usize];
            let node = &self.nodes[kd_node.node_idx as usize];

            // Use fast_distance for KD-Tree traversal and distance checking
            let dist = crate::distance::fast_distance(lat, lon, node.lat, node.lon);
            
            // fast_distance returns the result in kilometers, convert to meters
            if dist * 1000.0 <= radius_meters {
                result.push(kd_node.node_idx);
            }

            // Distance to the splitting axis
            let axis_dist = if lon_divides {
                crate::distance::fast_distance(lat, lon, lat, node.lon) * 1000.0
            } else {
                crate::distance::fast_distance(lat, lon, node.lat, lon) * 1000.0
            };

            let search_left = if lon_divides { lon < node.lon } else { lat < node.lat };

            if search_left {
                stack.push((kd_node.left_idx, !lon_divides));
                if axis_dist <= radius_meters {
                    stack.push((kd_node.right_idx, !lon_divides));
                }
            } else {
                stack.push((kd_node.right_idx, !lon_divides));
                if axis_dist <= radius_meters {
                    stack.push((kd_node.left_idx, !lon_divides));
                }
            }
        }

        result
    }
}
