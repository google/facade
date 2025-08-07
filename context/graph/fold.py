# Copyright 2025 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable
import random

# Type alias for node identifiers for clarity.
NodeType = Any

ZERO_WEIGHT_THRESHOLD = 1e-9
MAX_EDGES_FROM_MIDDLE = 1000


@dataclass(frozen=True)
class Edge:
    """
    Represents a directed edge in a bipartite graph.

    Attributes:
        i: Identifier of the origin node.
        j: Identifier of the destination node.
        is_middle: Whether the origin node `i` is a middle-type node.
        weight: The weight of the edge.
    """
    i: NodeType
    j: NodeType
    is_middle: bool
    weight: float


def two_hops_random_walk_neighbors(
    edges: Iterable[Edge],
    max_neighbors: int,
    max_edges_from_middle: int = MAX_EDGES_FROM_MIDDLE,
    weight_perturb_scale: float = 1e-8
) -> Dict[NodeType, List[Tuple[float, NodeType]]]:
    
    # 1. Filter edges with near-zero weight.
    filtered_edges = []
    for e in edges:
        if e.weight > ZERO_WEIGHT_THRESHOLD:
            filtered_edges.append(e)

    # 2. Group edges by origin and calculate the total weight for each origin node.
    # Also, check for consistent 'is_middle' status for each node.
    origin_totals = defaultdict(float)
    keyed_by_origin = defaultdict(list)
    node_is_middle_status = {}
    for edge in filtered_edges:
        origin_totals[edge.i] += edge.weight
        keyed_by_origin[edge.i].append(edge)
        # Check for consistent node type (middle vs. non-middle)
        if edge.i in node_is_middle_status and node_is_middle_status[edge.i] != edge.is_middle:
            raise ValueError(
                f"Node {edge.i} appears as both a middle and non-middle type."
            )
        node_is_middle_status[edge.i] = edge.is_middle
    
    # 3. Normalize weights and group by the middle node.
    # For a non-middle-to-middle edge (i->j), the data is keyed by j.
    # For a middle-to-non-middle edge (i->j), the data is keyed by i.
    grouped_by_middle_node = defaultdict(list)
    for origin, edge_list in keyed_by_origin.items():
        total_weight = origin_totals[origin]
        if total_weight == 0:
            continue
        
        for edge in edge_list:
            normalized_weight = edge.weight / total_weight
            if edge.is_middle:
                # Outgoing from middle node `i` to non-middle node `j`.
                # Key by middle node `i`, store (destination, weight, is_outgoing=True)
                grouped_by_middle_node[edge.i].append((edge.j, normalized_weight, True))
            else:
                # Incoming to middle node `j` from non-middle node `i`.
                # Key by middle node `j`, store (source, weight, is_outgoing=False)
                grouped_by_middle_node[edge.j].append((edge.i, normalized_weight, False))
    
    # 4. Compute 2-hop path weights via the cross-product at each middle node.
    path_weights = defaultdict(float)
    for middle_node, connections in grouped_by_middle_node.items():
        incoming_edges = []
        outgoing_edges = []
        for neighbor, weight, is_outgoing in connections:
            noise = random.uniform(0, weight_perturb_scale)
            if is_outgoing:
                outgoing_edges.append((neighbor, weight + noise))
            else:
                incoming_edges.append((neighbor, weight + noise))
        
        # Keep only the top N incoming and outgoing edges by weight.
        incoming_edges.sort(key=lambda x: x[1], reverse=True)
        outgoing_edges.sort(key=lambda x: x[1], reverse=True)

        top_incoming = incoming_edges[:max_edges_from_middle]
        top_outgoing = outgoing_edges[:max_edges_from_middle]

        # Calculate cross-product of weights for all 2-hop paths.
        for i_node, i_weight in top_incoming:
            for j_node, j_weight in top_outgoing:
                # Path is from i_node -> middle_node -> j_node
                path_weights[(i_node, j_node)] += i_weight * j_weight

    # 5. Group paths by the final destination node.
    neighbors_by_dest = defaultdict(list)
    for (start_node, end_node), weight in path_weights.items():
        neighbors_by_dest[end_node].append((weight, start_node))

    # 6. For each destination, get the top N neighbors sorted by weight.
    final_neighbors = {}
    for dest_node, neighbors in neighbors_by_dest.items():
        neighbors.sort(key=lambda x: x[0], reverse=True)
        final_neighbors[dest_node] = neighbors[:max_neighbors]

    return final_neighbors