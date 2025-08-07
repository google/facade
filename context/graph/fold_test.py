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

import unittest
from fold import Edge, two_hops_random_walk_neighbors
from typing import Dict, List, Tuple, Any

class TwoHopsRandomWalkNeighborsTest(unittest.TestCase):

    def assert_results_equal(self,
                             expected: Dict[Any, List[Tuple[float, Any]]],
                             actual: Dict[Any, List[Tuple[float, Any]]],
                             abs_tol: float = 1e-6):
        """
        Asserts that two result dictionaries are equal, handling unordered lists
        and floating point comparisons.
        """
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            # Sort lists of tuples for consistent comparison
            # The second element of the tuple (node ID) is used as a tie-breaker
            expected_neighbors = sorted(expected[key], key=lambda x: (-x[0], x[1]))
            actual_neighbors = sorted(actual[key], key=lambda x: (-x[0], x[1]))
            
            self.assertEqual(
                len(expected_neighbors),
                len(actual_neighbors),
                f"Mismatch in number of neighbors for key '{key}'"
            )
            for (ew, en), (aw, an) in zip(expected_neighbors, actual_neighbors):
                self.assertAlmostEqual(ew, aw, delta=abs_tol, msg=f"Weight mismatch for key '{key}'")
                self.assertEqual(en, an, msg=f"Node ID mismatch for key '{key}'")

    # @unittest.skip("")
    def test_unconnected_component_directed(self):
        """
        Tests a graph with two disconnected components where no 2-hop path exists.
        Graph: 0 -> 1*,  2* -> 3
        """
        edges = [
            Edge(i="0", j="1", is_middle=False, weight=1.0),
            Edge(i="2", j="3", is_middle=True, weight=1.0),
        ]
        result = two_hops_random_walk_neighbors(
            edges, max_neighbors=10, max_edges_from_middle=10, weight_perturb_scale=0.0
        )
        self.assertEqual(result, {})

    # @unittest.skip("")
    def test_unconnected_component_undirected(self):
        """
        Tests a graph with two disconnected self-loops.
        Graph: 0 - 1*,  2* - 3
        """
        edges = [
            Edge(i="0", j="1", is_middle=False, weight=1.0),
            Edge(i="1", j="0", is_middle=True, weight=1.0), # Reverse of first edge
            Edge(i="2", j="3", is_middle=True, weight=1.0),
            Edge(i="3", j="2", is_middle=False, weight=1.0), # Reverse of second edge
        ]
        result = two_hops_random_walk_neighbors(
            edges, max_neighbors=10, max_edges_from_middle=10, weight_perturb_scale=0.0
        )
        expected = {
            "0": [(1.0, "0")],
            "3": [(1.0, "3")],
        }
        self.assert_results_equal(expected, result)

    # @unittest.skip("")
    def test_simple_directed_graph(self):
        """
        Tests a simple directed graph with multiple paths.
        Graph: 0 -> 1*,  3 -> 1* -> 2,  3 -> 4* -> 5
        """
        edges = [
            Edge(i="0", j="1", is_middle=False, weight=1.0),
            Edge(i="3", j="1", is_middle=False, weight=2.0),
            Edge(i="1", j="2", is_middle=True, weight=3.0),
            Edge(i="3", j="4", is_middle=False, weight=5.0),
            Edge(i="4", j="5", is_middle=True, weight=7.0),
        ]
        result = two_hops_random_walk_neighbors(
            edges, max_neighbors=10, max_edges_from_middle=10, weight_perturb_scale=0.0
        )
        expected = {
            "2": [(1.0, "0"), (2.0 / 7.0, "3")],
            "5": [(5.0 / 7.0, "3")],
        }
        self.assert_results_equal(expected, result)
    
    # @unittest.skip("")
    def test_simple_undirected_graph(self):
        """
        Tests a more complex undirected graph.
        Graph: 
        0 - 1*,  
        3 - 1* - 2,  
        3 - 4* - 5
        """
        edges = [
            Edge(i="0", j="1", is_middle=False, weight=1.0),
            Edge(i="1", j="0", is_middle=True, weight=1.0),
            Edge(i="3", j="1", is_middle=False, weight=2.0),
            Edge(i="1", j="3", is_middle=True, weight=2.0),
            Edge(i="1", j="2", is_middle=True, weight=3.0),
            Edge(i="2", j="1", is_middle=False, weight=3.0),
            Edge(i="3", j="4", is_middle=False, weight=5.0),
            Edge(i="4", j="3", is_middle=True, weight=5.0),
            Edge(i="4", j="5", is_middle=True, weight=7.0),
            Edge(i="5", j="4", is_middle=False, weight=7.0),
        ]
        result = two_hops_random_walk_neighbors(
            edges, max_neighbors=10, max_edges_from_middle=10, weight_perturb_scale=0.0
        )
        # print("Result: ", result)
        # exit()
        expected = {
            "0": [(1.0/6.0, "0"), (1.0/6.0, "2"), (1.0/21.0, "3")],
            "2": [(0.5, "0"), (0.5, "2"), (1.0/7.0, "3")],
            "3": [(1.0/3.0, "0"), (1.0/3.0, "2"), (11.0/28.0, "3"), (5.0/12.0, "5")],
            "5": [(5.0/12.0, "3"), (7.0/12.0, "5")],
        }
        self.assert_results_equal(expected, result)
    
    # @unittest.skip("")
    def test_drops_zero_weight_edge(self):
        """
        Tests that an edge with zero weight is correctly dropped and does not
        contribute to any path.
        """
        edges = [
            Edge(i="0", j="1", is_middle=False, weight=0.0), # This should be dropped
            Edge(i="3", j="1", is_middle=False, weight=2.0),
            Edge(i="1", j="2", is_middle=True, weight=3.0),
            Edge(i="3", j="4", is_middle=False, weight=5.0),
            Edge(i="4", j="5", is_middle=True, weight=7.0),
        ]
        result = two_hops_random_walk_neighbors(
            edges, max_neighbors=10, max_edges_from_middle=10, weight_perturb_scale=0.0
        )
        expected = {
            "2": [(2.0 / 7.0, "3")],
            "5": [(5.0 / 7.0, "3")],
        }
        self.assert_results_equal(expected, result)

    # @unittest.skip("")
    def test_respects_max_neighbors(self):
        """
        Tests that the `max_neighbors` parameter is respected, returning only
        the top N neighbors for each destination node.
        """
        edges = [
            Edge(i="0", j="1", is_middle=False, weight=1.0), 
            Edge(i="1", j="0", is_middle=True, weight=1.0),
            Edge(i="3", j="1", is_middle=False, weight=2.0), 
            Edge(i="1", j="3", is_middle=True, weight=2.0),
            Edge(i="1", j="2", is_middle=True, weight=3.0), 
            Edge(i="2", j="1", is_middle=False, weight=3.0),
            Edge(i="3", j="4", is_middle=False, weight=5.0), 
            Edge(i="4", j="3", is_middle=True, weight=5.0),
            Edge(i="4", j="5", is_middle=True, weight=7.0), 
            Edge(i="5", j="4", is_middle=False, weight=7.0),
        ]
        result = two_hops_random_walk_neighbors(
            edges, max_neighbors=1, max_edges_from_middle=10, weight_perturb_scale=0.0
        )
        expected = {
            "0": [(1.0/6.0, "0")], #(1.0/6.0, "2") is also correct answer
            "2": [(0.5, "0")], #(0.5, "2") is also correct answer
            "3": [(5.0/12.0, "5")],
            "5": [(7.0/12.0, "5")],
        }
        self.assert_results_equal(expected, result)

    # @unittest.skip("")
    def test_respects_max_edges_from_middle_incoming(self):
        """
        Tests that `max_edges_from_middle` correctly prunes lower-weight
        incoming edges to a middle node.
        """
        edges = [
            Edge(i="0", j="1", is_middle=False, weight=1.0),   # From 0, total w=5.0
            Edge(i="0", j="10", is_middle=False, weight=4.0),
            Edge(i="2", j="1", is_middle=False, weight=3.0),   # From 2, total w=4.0
            Edge(i="2", j="10", is_middle=False, weight=1.0),
            Edge(i="3", j="1", is_middle=False, weight=3.0),   # From 3, total w=3.0
            Edge(i="1", j="5", is_middle=True, weight=1.0),
            Edge(i="10", j="4", is_middle=True, weight=1.0),
        ]
        # Middle node 1* gets incoming from 0 (w=1), 2 (w=3), 3 (w=3).
        # With max_edges_from_middle=2, it keeps edges from 2 and 3, dropping 0.
        result = two_hops_random_walk_neighbors(
            edges, max_neighbors=10, max_edges_from_middle=2, weight_perturb_scale=0.0
        )
        expected = {
            "4": [(4.0/5.0, "0"), (1.0/4.0, "2")],
            "5": [(1.0, "3"), (3.0/4.0, "2")],
        }
        self.assert_results_equal(expected, result)

    # @unittest.skip("")
    def test_respects_max_edges_from_middle_outgoing(self):
        """
        Tests that `max_edges_from_middle` correctly prunes lower-weight
        outgoing edges from a middle node.
        """
        edges = [
            Edge(i="1", j="0", is_middle=True, weight=1.0),
            Edge(i="1", j="2", is_middle=True, weight=3.0),
            Edge(i="1", j="3", is_middle=True, weight=2.0),
            Edge(i="4", j="10", is_middle=False, weight=1.0),
            Edge(i="5", j="1", is_middle=False, weight=1.0),
            Edge(i="10", j="0", is_middle=True, weight=4.0),
            Edge(i="10", j="2", is_middle=True, weight=1.0),
        ]
        # Middle node 1* has outgoing to 0 (w=1), 2 (w=3), 3 (w=2).
        # With max_edges_from_middle=2, it keeps edges to 2 and 3, dropping 0.
        # Total weight for normalization at 1* becomes 3+2=5.
        result = two_hops_random_walk_neighbors(
            edges, max_neighbors=10, max_edges_from_middle=2, weight_perturb_scale=0.0
        )
        expected = {
            "0": [(4.0/5.0, "4")],
            "2": [(3.0/6.0, "5"), (1.0/5.0, "4")], # C++ result has 3.0/(1+2+3) which is 3.0/6.0
            "3": [(2.0/6.0, "5")], # C++ result has 2.0/(1+2+3) which is 2.0/6.0
        }
        self.assert_results_equal(expected, result)

    # @unittest.skip("")
    def test_dies_on_inconsistent_middle_type(self):
        """
        Tests that the function raises a ValueError when a node is defined
        with conflicting `is_middle` statuses.
        Graph: 0 -> 1*, 0* -> 2
        """
        edges = [
            Edge(i="0", j="1", is_middle=False, weight=1.0),
            Edge(i="0", j="2", is_middle=True, weight=1.0),
        ]
        with self.assertRaisesRegex(ValueError, "appears as both a middle and non-middle type"):
            two_hops_random_walk_neighbors(
                edges, max_neighbors=10, max_edges_from_middle=10, weight_perturb_scale=0.0
            )

if __name__ == "__main__":
    unittest.main()