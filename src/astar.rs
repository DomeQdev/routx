// (c) Copyright 2025 Mikołaj Kuranowski
// SPDX-License-Identifier: MIT

use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

use crate::{earth_distance, Edge, Graph, Node};

/// Recommended number of allowed node expansions in [find_route](crate::find_route) and
/// [find_route_without_turn_around](crate::find_route_without_turn_around)
/// before [AStarError::StepLimitExceeded] is returned.
pub const DEFAULT_STEP_LIMIT: usize = 1_000_000;

/// Error conditions which may occur during [find_route](crate::find_route) or
/// [find_route_without_turn_around](crate::find_route_without_turn_around).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AStarError {
    /// The start or end nodes don't exist in a graph.
    InvalidReference(i64),

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
            Self::InvalidReference(node_id) => write!(f, "invalid node: {}", node_id),
            Self::StepLimitExceeded => write!(f, "step limit exceeded"),
        }
    }
}

impl std::error::Error for AStarError {}

/// SearchNode represents a node in the A* search space.
///
/// For [find_route], the search space nodes coincide with [graph nodes](crate::Node) -
/// hence this trait is implemented for [i64], a simple node id reference.
///
/// For [find_route_without_turn_around], the search space contains not only the current
/// [graph node](crate::Node), but also the predecessor's osm_id. This comes from practicality,
/// in order to prevent A-B-A turn-around reaching node B from node A is distinct from
/// reaching node B from node C -- the former prohibits reaching node A, while the latter does not.
/// This case is handled by implementing this trait for [NodeAndBefore].
trait SearchNode: Sized + Eq + Copy + Hash {
    fn initial(node_id: i64) -> Self;
    fn node_id(&self) -> i64;
    fn next(&self, n: &Node, at_osm_id: i64) -> Option<Self>;
}

impl SearchNode for i64 {
    fn initial(node_id: i64) -> Self {
        node_id
    }

    fn node_id(&self) -> i64 {
        *self
    }

    fn next(&self, n: &Node, _at_osm_id: i64) -> Option<Self> {
        Some(n.id)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct NodeAndBefore {
    node_id: i64,
    before_osm_id: i64,
}

impl SearchNode for NodeAndBefore {
    fn initial(node_id: i64) -> Self {
        Self {
            node_id,
            before_osm_id: 0,
        }
    }

    fn node_id(&self) -> i64 {
        self.node_id
    }

    fn next(&self, n: &Node, at_osm_id: i64) -> Option<Self> {
        if n.osm_id == self.before_osm_id {
            // Disallow A-B-A immediate turn-around
            None
        } else {
            Some(Self {
                node_id: n.id,
                before_osm_id: at_osm_id,
            })
        }
    }
}

/// Represents an element stored in the A* expansion queue. The [Eq] and [Ord] traits
/// delegate to comparing scores only (and in reverse order, to get a min-heap instead of
/// Rust's default max-heap), as this is what matters for the A* Queue's order.
struct QueueItem<N: SearchNode> {
    /// Current node in the search space
    at: N,

    /// osm_id of the current node
    at_osm_id: i64,

    /// Cost required to reach [QueueItem::at]
    cost: f32,

    /// A* order heuristic, cost + earth_distance between [QueueItem::at] and the destination.
    /// Lower bound on the cost of the entire journey.
    score: f32,
}

impl<N: SearchNode> PartialEq for QueueItem<N> {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl<N: SearchNode> Eq for QueueItem<N> {}

impl<N: SearchNode> PartialOrd for QueueItem<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<N: SearchNode> Ord for QueueItem<N> {
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

fn reconstruct_path<N: SearchNode>(came_from: &HashMap<N, N>, mut last: N) -> Vec<i64> {
    let mut path = vec![last.node_id()];

    while let Some(&nd) = came_from.get(&last) {
        path.push(nd.node_id());
        last = nd;
    }

    path.reverse();
    return path;
}

fn find_route_inner<N: SearchNode>(
    g: &Graph,
    from_id: i64,
    to_id: i64,
    step_limit: usize,
) -> Result<Vec<i64>, AStarError> {
    assert_ne!(from_id, 0);
    assert_ne!(to_id, 0);

    let mut queue: BinaryHeap<QueueItem<N>> = BinaryHeap::default();
    let mut came_from: HashMap<N, N> = HashMap::default();
    let mut known_costs: HashMap<N, f32> = HashMap::default();
    let mut steps: usize = 0;

    let to_node = g
        .get_node(to_id)
        .ok_or(AStarError::InvalidReference(to_id))?;

    // Push the `from` node to the queue
    {
        let initial = N::initial(from_id);

        let from_node = g
            .get_node(from_id)
            .ok_or(AStarError::InvalidReference(from_id))?;

        queue.push(QueueItem::<N> {
            at: initial,
            at_osm_id: from_node.osm_id,
            cost: 0.0,
            score: earth_distance(from_node.lat, from_node.lon, to_node.lat, to_node.lon),
        });
        known_costs.insert(initial, 0.0);
    }

    // Expand elements from the queue until either the `to` node is reached,
    // or the step limit is exceeded
    while let Some(item) = queue.pop() {
        // Check if destination was reached
        if item.at_osm_id == to_id {
            return Ok(reconstruct_path(&mut came_from, item.at));
        }

        // Contrary to the wikipedia definition, we might keep multiple items in the queue for the same node.
        if item.cost > known_costs.get(&item.at).cloned().unwrap_or(f32::INFINITY) {
            continue;
        }

        // Check against the step limit
        steps += 1;
        if steps > step_limit {
            return Err(AStarError::StepLimitExceeded);
        }

        // Expand the node by adding its neighbors to the queue
        for &Edge {
            to: neighbor_id,
            cost: edge_cost,
        } in g.get_edges(item.at.node_id())
        {
            assert_ne!(neighbor_id, 0);

            // Ensure the referenced neighbor_id exists
            let neighbor = match g.get_node(neighbor_id) {
                Some(n) => n,
                None => continue,
            };

            // Check if the search space allows advancing to this neighbor
            let neighbor_at = match item.at.next(&neighbor, item.at_osm_id) {
                Some(n) => n,
                None => continue,
            };

            // Check if this is the cheapest way to the neighbor
            let neighbor_cost = item.cost + edge_cost;
            if neighbor_cost
                > known_costs
                    .get(&neighbor_at)
                    .cloned()
                    .unwrap_or(f32::INFINITY)
            {
                continue;
            }

            // Push the new item into the queue
            came_from.insert(neighbor_at, item.at);
            known_costs.insert(neighbor_at, neighbor_cost);
            queue.push(QueueItem::<N> {
                at: neighbor_at,
                at_osm_id: neighbor.osm_id,
                cost: neighbor_cost,
                score: neighbor_cost
                    + earth_distance(neighbor.lat, neighbor.lon, to_node.lat, to_node.lon),
            });
        }
    }

    Ok(vec![])
}

/// Uses the [A* algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
/// to find the shortest route between two nodes in the provided graph.
///
/// Returns an empty vector if there is no route between the two nodes.
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
    g: &Graph,
    from_id: i64,
    to_id: i64,
    step_limit: usize,
) -> Result<Vec<i64>, AStarError> {
    find_route_inner::<i64>(g, from_id, to_id, step_limit)
}

/// Uses the [A* algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
/// to find the shortest route between two points in the provided graph.
///
/// Returns an empty list if there is no route between the two points.
///
/// For graphs without turn restrictions, use [find_route](super::find_route) as it runs faster.
/// This function has an extra dimension - it needs to not only consider the current node,
/// but also what was the previous node to prevent A-B-A immediate turnaround instructions.
///
/// `step_limit` limits how many nodes may be expanded during the search
/// before returning [AStarError::StepLimitExceeded]. Concluding that no route exists requires
/// expanding all nodes accessible from the start, which is usually very time-consuming,
/// especially on large datasets (like the whole planet). The recommended value is
/// [DEFAULT_STEP_LIMIT](crate::DEFAULT_STEP_LIMIT).
pub fn find_route_without_turn_around(
    g: &Graph,
    from_id: i64,
    to_id: i64,
    step_limit: usize,
) -> Result<Vec<i64>, AStarError> {
    find_route_inner::<NodeAndBefore>(g, from_id, to_id, step_limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Node};

    #[inline]
    fn simple_graph_fixture() -> Graph {
        //   200   200   200
        // 1─────2─────3─────4
        //       └─────5─────┘
        //         100    100
        Graph::from_iter(
            [
                Node {
                    id: 1,
                    osm_id: 1,
                    lat: 0.01,
                    lon: 0.01,
                },
                Node {
                    id: 2,
                    osm_id: 2,
                    lat: 0.02,
                    lon: 0.01,
                },
                Node {
                    id: 3,
                    osm_id: 3,
                    lat: 0.03,
                    lon: 0.01,
                },
                Node {
                    id: 4,
                    osm_id: 4,
                    lat: 0.04,
                    lon: 0.01,
                },
                Node {
                    id: 5,
                    osm_id: 5,
                    lat: 0.03,
                    lon: 0.00,
                },
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
        assert_eq!(find_route(&g, 1, 4, 100), Ok(vec![1_i64, 2, 5, 4]));
    }

    #[test]
    fn simple_without_turn_around() {
        let g = simple_graph_fixture();
        assert_eq!(
            find_route_without_turn_around(&g, 1, 4, 100),
            Ok(vec![1_i64, 2, 5, 4])
        );
    }

    #[test]
    fn step_limit() {
        let g = simple_graph_fixture();
        assert_eq!(find_route(&g, 1, 4, 2), Err(AStarError::StepLimitExceeded));
    }

    #[test]
    fn step_limit_without_turn_around() {
        let g = simple_graph_fixture();
        assert_eq!(
            find_route_without_turn_around(&g, 1, 4, 2),
            Err(AStarError::StepLimitExceeded)
        );
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
                Node {
                    id: 1,
                    osm_id: 1,
                    lat: 0.00,
                    lon: 0.00,
                },
                Node {
                    id: 2,
                    osm_id: 2,
                    lat: 0.01,
                    lon: 0.00,
                },
                Node {
                    id: 3,
                    osm_id: 3,
                    lat: 0.02,
                    lon: 0.00,
                },
                Node {
                    id: 4,
                    osm_id: 4,
                    lat: 0.00,
                    lon: 0.01,
                },
                Node {
                    id: 5,
                    osm_id: 5,
                    lat: 0.01,
                    lon: 0.01,
                },
                Node {
                    id: 6,
                    osm_id: 6,
                    lat: 0.02,
                    lon: 0.01,
                },
                Node {
                    id: 7,
                    osm_id: 7,
                    lat: 0.00,
                    lon: 0.02,
                },
                Node {
                    id: 8,
                    osm_id: 8,
                    lat: 0.01,
                    lon: 0.02,
                },
                Node {
                    id: 9,
                    osm_id: 9,
                    lat: 0.02,
                    lon: 0.02,
                },
            ],
            [
                (1, 2, 100.0),
                (1, 4, 600.0),
                (2, 1, 100.0),
                (2, 3, 200.0),
                (2, 5, 500.0),
                (3, 2, 200.0),
                (3, 6, 100.0),
                (4, 1, 600.0),
                (4, 5, 200.0),
                (4, 7, 400.0),
                (5, 2, 500.0),
                (5, 4, 200.0),
                (5, 6, 400.0),
                (5, 8, 300.0),
                (6, 3, 100.0),
                (6, 5, 400.0),
                (6, 9, 100.0),
                (7, 4, 400.0),
                (7, 8, 500.0),
                (8, 5, 300.0),
                (8, 7, 500.0),
                (8, 9, 100.0),
                (9, 6, 100.0),
                (9, 8, 100.0),
            ],
        )
    }

    #[test]
    fn shortest_not_optimal() {
        let g = shortest_not_optimal_fixture();
        assert_eq!(find_route(&g, 1, 8, 100), Ok(vec![1_i64, 2, 3, 6, 9, 8]));
    }

    #[test]
    fn shortest_not_optimal_without_turn_around() {
        let g = shortest_not_optimal_fixture();
        assert_eq!(
            find_route_without_turn_around(&g, 1, 8, 100),
            Ok(vec![1_i64, 2, 3, 6, 9, 8])
        );
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
                Node {
                    id: 1,
                    osm_id: 1,
                    lat: 0.00,
                    lon: 0.02,
                },
                Node {
                    id: 2,
                    osm_id: 2,
                    lat: 0.00,
                    lon: 0.01,
                },
                Node {
                    id: 20,
                    osm_id: 2,
                    lat: 0.00,
                    lon: 0.01,
                },
                Node {
                    id: 3,
                    osm_id: 3,
                    lat: 0.00,
                    lon: 0.00,
                },
                Node {
                    id: 4,
                    osm_id: 4,
                    lat: 0.01,
                    lon: 0.01,
                },
                Node {
                    id: 5,
                    osm_id: 5,
                    lat: 0.01,
                    lon: 0.00,
                },
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
        assert_eq!(find_route(&g, 1, 3, 100), Ok(vec![1_i64, 20, 4, 2, 3]));
    }

    #[test]
    fn turn_restriction_without_turn_around() {
        let g = turn_restriction_fixture();
        assert_eq!(
            find_route_without_turn_around(&g, 1, 3, 100),
            Ok(vec![1_i64, 20, 4, 5, 3])
        );
    }
}
