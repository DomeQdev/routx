// (c) Copyright 2025 Mikołaj Kuranowski
// SPDX-License-Identifier: MIT

use std::collections::BinaryHeap;

use crate::earth_distance;
use crate::flat_graph::{MemoryMappedGraph, NULL_IDX};

/// Recommended number of allowed node expansions in [find_route](crate::find_route) and
/// [find_route_without_turn_around](crate::find_route_without_turn_around)
/// before [AStarError::StepLimitExceeded] is returned.
pub const DEFAULT_STEP_LIMIT: usize = 1_000_000;

/// Error conditions which may occur during [find_route](crate::find_route) or
/// [find_route_without_turn_around](crate::find_route_without_turn_around).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AStarError {
    /// The start or end nodes don't exist in a graph.
    InvalidReference(u32),

    /// Route search has exceeded its limit of steps.
    /// Either the nodes are really far apart, or no route exists.
    ///
    /// Concluding that no route exists requires traversing the whole graph,
    /// which can result in a denial-of-service. The step limit protects
    /// against resource exhaustion.
    StepLimitExceeded,
}

impl std::fmt::Display for AStarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidReference(node_idx) => write!(f, "invalid node index: {}", node_idx),
            Self::StepLimitExceeded => write!(f, "step limit exceeded"),
        }
    }
}

impl std::error::Error for AStarError {}

/// Represents an element stored in the A* expansion queue. The [Eq] and [Ord] traits
/// delegate to comparing scores only (and in reverse order, to get a min-heap instead of
/// Rust's default max-heap), as this is what matters for the A* Queue's order.
struct QueueItem {
    /// Current node index in the memory-mapped graph
    node_idx: u32,

    /// osm_id of the previous node (used to prevent A-B-A turnarounds)
    prev_osm_id: i64,

    /// Cost required to reach [QueueItem::node_idx]
    cost: f32,

    /// A* order heuristic, cost + earth_distance between [QueueItem::node_idx] and the destination.
    /// Lower bound on the cost of the entire journey.
    score: f32,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // NOTE: We revert the order of comparison,
        // as lower scores are considered better ("higher"),
        // and Rust's BinaryHeap is a max-heap.
        other
            .score
            .partial_cmp(&self.score)
            .expect("A* QueueItem's score must be comparable")
    }
}

fn reconstruct_path(came_from: &[u32], mut current: u32) -> Vec<u32> {
    let mut path = vec![current];

    while came_from[current as usize] != NULL_IDX {
        current = came_from[current as usize];
        path.push(current);
    }

    path.reverse();
    path
}

fn find_route_inner(
    g: &MemoryMappedGraph,
    from_idx: u32,
    to_idx: u32,
    step_limit: usize,
    prevent_u_turns: bool,
) -> Result<Vec<u32>, AStarError> {
    let nodes_count = g.nodes_count() as usize;

    if from_idx as usize >= nodes_count {
        return Err(AStarError::InvalidReference(from_idx));
    }
    if to_idx as usize >= nodes_count {
        return Err(AStarError::InvalidReference(to_idx));
    }

    let to_node = g.get_node(to_idx).unwrap();
    let target_osm_id = to_node.osm_id;
    let target_lat = to_node.lat;
    let target_lon = to_node.lon;

    let from_node = g.get_node(from_idx).unwrap();

    let mut known_costs = vec![f32::INFINITY; nodes_count];
    let mut came_from = vec![NULL_IDX; nodes_count];
    
    let mut queue: BinaryHeap<QueueItem> = BinaryHeap::new();
    let mut steps: usize = 0;

    // Push the `from` node to the queue
    known_costs[from_idx as usize] = 0.0;
    queue.push(QueueItem {
        node_idx: from_idx,
        prev_osm_id: 0,
        cost: 0.0,
        score: earth_distance(from_node.lat, from_node.lon, target_lat, target_lon),
    });

    // Expand elements from the queue until either the `to` node is reached,
    // or the step limit is exceeded
    while let Some(item) = queue.pop() {
        let curr_node = g.get_node(item.node_idx).unwrap();

        // Check if destination was reached.
        // We match by OSM ID because turn restrictions might have created multiple phantom nodes
        // representing the same physical location.
        if curr_node.osm_id == target_osm_id {
            return Ok(reconstruct_path(&came_from, item.node_idx));
        }

        // We might keep multiple items in the queue for the same node.
        if item.cost > known_costs[item.node_idx as usize] {
            continue;
        }

        // Check against the step limit
        steps += 1;
        if steps > step_limit {
            return Err(AStarError::StepLimitExceeded);
        }

        // Expand the node by adding its neighbors to the queue
        for edge in g.get_edges(curr_node) {
            let neighbor_idx = edge.to_node_idx;
            
            // Ensure the referenced neighbor_idx exists (should always be true for generated graph)
            let neighbor = match g.get_node(neighbor_idx) {
                Some(n) => n,
                None => continue,
            };

            // Prevent immediate A-B-A turnarounds if requested
            if prevent_u_turns && neighbor.osm_id == item.prev_osm_id {
                continue;
            }

            // Check if this is the cheapest way to the neighbor
            let new_cost = item.cost + edge.cost;
            if new_cost < known_costs[neighbor_idx as usize] {
                known_costs[neighbor_idx as usize] = new_cost;
                came_from[neighbor_idx as usize] = item.node_idx;
                
                queue.push(QueueItem {
                    node_idx: neighbor_idx,
                    prev_osm_id: curr_node.osm_id,
                    cost: new_cost,
                    score: new_cost + earth_distance(neighbor.lat, neighbor.lon, target_lat, target_lon),
                });
            }
        }
    }

    Ok(vec![])
}

/// Uses the [A* algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
/// to find the shortest route between two nodes in the provided graph.
///
/// Returns an empty vector if there is no route between the two nodes.
///
/// `from_idx` and `to_idx` must identify a specific valid Node internal index
/// in the [MemoryMappedGraph]; otherwise [AStarError::InvalidReference] is returned.
///
/// The returned route may end at a different, non-canonical node, as long as its
/// [osm_id](crate::flat_graph::FlatNode::osm_id) is equal to the target node's `osm_id`.
///
/// For graphs with turn restrictions, use [find_route_without_turn_around](super::find_route_without_turn_around),
/// as this implementation will generate instructions with immediate turnarounds
/// (A-B-A) to circumvent any restrictions.
///
/// `step_limit` limits how many nodes may be expanded during the search
/// before returning [AStarError::StepLimitExceeded]. Concluding that no route exists requires
/// expanding all nodes accessible from the start, which is usually very time-consuming,
/// especially on large datasets (like the whole planet). The recommended value is
/// [DEFAULT_STEP_LIMIT](crate::DEFAULT_STEP_LIMIT).
pub fn find_route(
    g: &MemoryMappedGraph,
    from_idx: u32,
    to_idx: u32,
    step_limit: usize,
) -> Result<Vec<u32>, AStarError> {
    find_route_inner(g, from_idx, to_idx, step_limit, false)
}

/// Uses the [A* algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
/// to find the shortest route between two points in the provided graph.
///
/// Returns an empty list if there is no route between the two points.
///
/// `from_idx` and `to_idx` must identify a specific valid Node internal index
/// in the [MemoryMappedGraph]; otherwise [AStarError::InvalidReference] is returned.
///
/// The returned route may end at a different, non-canonical node, as long as its
/// [osm_id](crate::flat_graph::FlatNode::osm_id) is equal to the target node's `osm_id`.
///
/// For graphs without turn restrictions, use [find_route](super::find_route) as it runs faster.
/// This function has an extra dimension - it prevents A-B-A immediate turnaround instructions.
///
/// `step_limit` limits how many nodes may be expanded during the search
/// before returning [AStarError::StepLimitExceeded]. Concluding that no route exists requires
/// expanding all nodes accessible from the start, which is usually very time-consuming,
/// especially on large datasets (like the whole planet). The recommended value is
/// [DEFAULT_STEP_LIMIT](crate::DEFAULT_STEP_LIMIT).
pub fn find_route_without_turn_around(
    g: &MemoryMappedGraph,
    from_idx: u32,
    to_idx: u32,
    step_limit: usize,
) -> Result<Vec<u32>, AStarError> {
    find_route_inner(g, from_idx, to_idx, step_limit, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Node};
    use crate::builder::build_memory_mapped_graph;
    use std::io::Cursor;

    fn build_test_mmap(g: &Graph) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        build_memory_mapped_graph(g, None, &mut buf).unwrap();
        buf.into_inner()
    }

    fn find_idx_by_osm_id(mmap: &MemoryMappedGraph, osm_id: i64) -> u32 {
        for i in 0..mmap.nodes_count() {
            if mmap.get_node(i).unwrap().osm_id == osm_id {
                return i;
            }
        }
        panic!("Node with osm_id {} not found", osm_id);
    }

    fn get_osm_path(mmap: &MemoryMappedGraph, path: &[u32]) -> Vec<i64> {
        path.iter().map(|&idx| mmap.get_node(idx).unwrap().osm_id).collect()
    }

    #[inline]
    fn simple_graph_fixture() -> Graph {
        //   200   200   200
        // 1─────2─────3─────4
        //       └─────5─────┘
        //         100    100
        Graph::from_iter(
            [
                Node { id: 1, osm_id: 1, lat: 0.01, lon: 0.01 },
                Node { id: 2, osm_id: 2, lat: 0.02, lon: 0.01 },
                Node { id: 3, osm_id: 3, lat: 0.03, lon: 0.01 },
                Node { id: 4, osm_id: 4, lat: 0.04, lon: 0.01 },
                Node { id: 5, osm_id: 5, lat: 0.03, lon: 0.00 },
            ],
            [
                (1, 2, 200.0),
                (2, 1, 200.0),
                (2, 3, 200.0),
                (2, 5, 100.0),
                (3, 2, 200.0),
                (3, 4, 200.0),
                (4, 3, 200.0),
                (4, 5, 100.0),
                (5, 2, 100.0),
                (5, 4, 100.0),
            ],
        )
    }

    #[test]
    fn simple() {
        let g = simple_graph_fixture();
        let buf = build_test_mmap(&g);
        let mmap = MemoryMappedGraph::new(&buf).unwrap();

        let from = find_idx_by_osm_id(&mmap, 1);
        let to = find_idx_by_osm_id(&mmap, 4);

        let route = find_route(&mmap, from, to, 100).unwrap();
        assert_eq!(get_osm_path(&mmap, &route), vec![1_i64, 2, 5, 4]);
    }

    #[test]
    fn simple_without_turn_around() {
        let g = simple_graph_fixture();
        let buf = build_test_mmap(&g);
        let mmap = MemoryMappedGraph::new(&buf).unwrap();

        let from = find_idx_by_osm_id(&mmap, 1);
        let to = find_idx_by_osm_id(&mmap, 4);

        let route = find_route_without_turn_around(&mmap, from, to, 100).unwrap();
        assert_eq!(get_osm_path(&mmap, &route), vec![1_i64, 2, 5, 4]);
    }

    #[test]
    fn step_limit() {
        let g = simple_graph_fixture();
        let buf = build_test_mmap(&g);
        let mmap = MemoryMappedGraph::new(&buf).unwrap();

        let from = find_idx_by_osm_id(&mmap, 1);
        let to = find_idx_by_osm_id(&mmap, 4);

        assert_eq!(find_route(&mmap, from, to, 2), Err(AStarError::StepLimitExceeded));
    }

    #[inline]
    fn shortest_not_optimal_fixture() -> Graph {
        //    500   100
        //  7─────8─────9
        //  │     │     │
        //  │400  │300  │100
        //  │ 200 │ 400 │
        //  4─────5─────6
        //  │     │     │
        //  │600  │500  │100
        //  │ 100 │ 200 │
        //  1─────2─────3
        Graph::from_iter(
            [
                Node { id: 1, osm_id: 1, lat: 0.00, lon: 0.00 },
                Node { id: 2, osm_id: 2, lat: 0.01, lon: 0.00 },
                Node { id: 3, osm_id: 3, lat: 0.02, lon: 0.00 },
                Node { id: 4, osm_id: 4, lat: 0.00, lon: 0.01 },
                Node { id: 5, osm_id: 5, lat: 0.01, lon: 0.01 },
                Node { id: 6, osm_id: 6, lat: 0.02, lon: 0.01 },
                Node { id: 7, osm_id: 7, lat: 0.00, lon: 0.02 },
                Node { id: 8, osm_id: 8, lat: 0.01, lon: 0.02 },
                Node { id: 9, osm_id: 9, lat: 0.02, lon: 0.02 },
            ],
            [
                (1, 2, 100.0), (1, 4, 600.0),
                (2, 1, 100.0), (2, 3, 200.0), (2, 5, 500.0),
                (3, 2, 200.0), (3, 6, 100.0),
                (4, 1, 600.0), (4, 5, 200.0), (4, 7, 400.0),
                (5, 2, 500.0), (5, 4, 200.0), (5, 6, 400.0), (5, 8, 300.0),
                (6, 3, 100.0), (6, 5, 400.0), (6, 9, 100.0),
                (7, 4, 400.0), (7, 8, 500.0),
                (8, 5, 300.0), (8, 7, 500.0), (8, 9, 100.0),
                (9, 6, 100.0), (9, 8, 100.0),
            ],
        )
    }

    #[test]
    fn shortest_not_optimal() {
        let g = shortest_not_optimal_fixture();
        let buf = build_test_mmap(&g);
        let mmap = MemoryMappedGraph::new(&buf).unwrap();

        let from = find_idx_by_osm_id(&mmap, 1);
        let to = find_idx_by_osm_id(&mmap, 8);

        let route = find_route(&mmap, from, to, 100).unwrap();
        assert_eq!(get_osm_path(&mmap, &route), vec![1_i64, 2, 3, 6, 9, 8]);
    }

    #[inline]
    fn turn_restriction_fixture() -> Graph {
        // 1
        // │
        // │10
        // │ 10
        // 2─────4
        // │     │
        // │10   │100
        // │ 10  │
        // 3─────5
        // mandatory 1-2-4
        Graph::from_iter(
            [
                Node { id: 1, osm_id: 1, lat: 0.00, lon: 0.02 },
                Node { id: 2, osm_id: 2, lat: 0.00, lon: 0.01 },
                Node { id: 20, osm_id: 2, lat: 0.00, lon: 0.01 },
                Node { id: 3, osm_id: 3, lat: 0.00, lon: 0.00 },
                Node { id: 4, osm_id: 4, lat: 0.01, lon: 0.01 },
                Node { id: 5, osm_id: 5, lat: 0.01, lon: 0.00 },
            ],
            [
                (1, 20, 10.0),
                (2, 1, 10.0),
                (2, 3, 10.0),
                (2, 4, 10.0),
                (20, 4, 10.0),
                (3, 2, 10.0),
                (3, 5, 10.0),
                (4, 2, 10.0),
                (4, 5, 100.0),
                (5, 3, 10.0),
                (5, 4, 100.0),
            ],
        )
    }

    #[test]
    fn turn_restriction() {
        let g = turn_restriction_fixture();
        let buf = build_test_mmap(&g);
        let mmap = MemoryMappedGraph::new(&buf).unwrap();

        // 1 -> 3
        let from = find_idx_by_osm_id(&mmap, 1);
        let to = find_idx_by_osm_id(&mmap, 3); // We want to reach OSM_ID = 3
        
        let route = find_route(&mmap, from, to, 100).unwrap();
        // Check the OSM ID path to ensure we bypass the restriction (via phantom nodes)
        // 1 -> 2 (phantom 20) -> 4 -> 2 -> 3
        assert_eq!(get_osm_path(&mmap, &route), vec![1_i64, 2, 4, 2, 3]);
    }

    #[test]
    fn turn_restriction_without_turn_around() {
        let g = turn_restriction_fixture();
        let buf = build_test_mmap(&g);
        let mmap = MemoryMappedGraph::new(&buf).unwrap();

        let from = find_idx_by_osm_id(&mmap, 1);
        let to = find_idx_by_osm_id(&mmap, 3);
        
        let route = find_route_without_turn_around(&mmap, from, to, 100).unwrap();
        // A-B-A is blocked, so it goes around: 1 -> 2(20) -> 4 -> 5 -> 3
        assert_eq!(get_osm_path(&mmap, &route), vec![1_i64, 2, 4, 5, 3]);
    }
}
