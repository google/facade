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
import random
from typing import List
from pipelines.history_featurizer.online_heaviest_items import ItemWeight, PositiveWeightedItemStore, OnlineHeaviestItems


def iw_match(test_case: unittest.TestCase, iw: ItemWeight, expected_item: str, expected_weight: float):
    test_case.assertEqual(iw.item, expected_item)
    test_case.assertAlmostEqual(iw.weight, expected_weight)


def contents_match(test_case: unittest.TestCase, elements: List[ItemWeight], expected: List[tuple[str, float]], ordered=True):
    test_case.assertEqual(len(elements), len(expected))
    if ordered:
        for i in range(len(elements)):
            iw_match(test_case, elements[i], expected[i][0], expected[i][1])
    else:
        sorted_elements = sorted(elements, key=lambda x: x.item)
        sorted_expected = sorted(expected, key=lambda x: x[0])
        for i in range(len(sorted_elements)):
            iw_match(test_case, sorted_elements[i], sorted_expected[i][0], sorted_expected[i][1])


class TestPositiveWeightedItemStore(unittest.TestCase):

    
    def test_accumulates(self):
        s = PositiveWeightedItemStore(is_min_heap=True)
        self.assertEqual(s.contents(), [])
        s.upsert("a", 1)
        contents_match(self, s.contents(), [("a", 1)])
        s.upsert("b", 2)
        contents_match(self, s.contents(), [("a", 1), ("b", 2)], ordered=False) # Order not guaranteed before heapify
        # Handles positive weight update.
        s.upsert("a", 4)
        # Find item 'a' and check its weight
        found = False
        for iw in s.contents():
            if iw.item == "a":
                self.assertAlmostEqual(iw.weight, 5)
                found = True
        self.assertTrue(found)

        # Handles negative weight update.
        s.upsert("b", -1)
        found = False
        for iw in s.contents():
            if iw.item == "b":
                self.assertAlmostEqual(iw.weight, 1)
                found = True
        self.assertTrue(found)

    
    def test_update_only(self):
        s = PositiveWeightedItemStore(is_min_heap=True)
        s.upsert("a", 1)

        # Disallow new element.
        self.assertFalse(s.upsert("b", 2, allow_insertion=False))
        contents_match(self, s.contents(), [("a", 1)])

        # Disallow new element, even if weight is negative.
        self.assertFalse(s.upsert("b", 0, allow_insertion=False))
        contents_match(self, s.contents(), [("a", 1)])

        # Updates existing element.
        self.assertTrue(s.upsert("a", 3, allow_insertion=False))
        contents_match(self, s.contents(), [("a", 4)])
        self.assertTrue(s.upsert("a", -1, allow_insertion=False))
        contents_match(self, s.contents(), [("a", 3)])

    
    def test_drops_non_positive_weights(self):
        s = PositiveWeightedItemStore(is_min_heap=True)
        # No weight update for initially non-positive elements
        s.upsert("a", 0)
        s.upsert("b", -1)
        self.assertEqual(s.contents(), [])
        s.upsert("a", 1)
        s.upsert("b", 1)
        s.upsert("c", 1)
        self.assertEqual(s.size(), 3)

        # Check that dropping an element works at any of the 3 positions
        # First
        s.upsert("a", -1) # Drop 'a'
        self.assertEqual(s.size(), 2)
        self.assertNotIn("a", s._index)
        s.upsert("a", 1)
        self.assertEqual(s.size(), 3)

        # Middle
        s.upsert("b", -2) # Drop 'b'
        self.assertEqual(s.size(), 2)
        self.assertNotIn("b", s._index)
        s.upsert("b", 1)
        self.assertEqual(s.size(), 3)

        # Last
        s.upsert("b", -4)
        self.assertEqual(s.size(), 2)
        self.assertNotIn("b", s._index)
        s.upsert("b", 1)
        self.assertEqual(s.size(), 3)

    
    def test_peek_pop_fails_empty(self):
        s = PositiveWeightedItemStore(is_min_heap=True)
        with self.assertRaises(IndexError):
            s.peek()
        with self.assertRaises(IndexError):
            s.pop()

    
    def test_max_heap(self):
        s = PositiveWeightedItemStore(is_min_heap=False)
        s.upsert("a", 8)
        s.upsert("b", 4)
        s.upsert("c", 5)
        s.upsert("d", 3)
        s.upsert("e", 9)
        s.upsert("f", 1)
        # Python heap is maintained on upsert
        expected = sorted([("a", 8), ("b", 4), ("c", 5), ("d", 3), ("e", 9), ("f", 1)], key=lambda x: x[1], reverse=True)
        popped = []
        while s.size() > 0:
            popped.append(s.pop())
        contents_match(self, popped, expected, ordered=True)

    
    def test_min_heap(self):
        s = PositiveWeightedItemStore(is_min_heap=True)
        s.upsert("a", 8)
        s.upsert("b", 4)
        s.upsert("c", 5)
        s.upsert("d", 3)
        s.upsert("e", 9)
        s.upsert("f", 1)
        expected = sorted([("a", 8), ("b", 4), ("c", 5), ("d", 3), ("e", 9), ("f", 1)], key=lambda x: x[1])
        popped = []
        while s.size() > 0:
            popped.append(s.pop())
        contents_match(self, popped, expected, ordered=True)

    
    def test_updates_weights_min_heap(self):
        s = PositiveWeightedItemStore(is_min_heap=True)
        s.upsert("a", 8)
        s.upsert("b", 4)
        s.upsert("c", 5)
        s.upsert("d", 3)
        s.upsert("e", 9)
        s.upsert("f", 1)

        iw_match(self, s.peek(), "f", 1)
        s.upsert("d", -2)  # No order change because 1 <= 1.
        iw_match(self, s.peek(), "f", 1) # f still smaller
        s.upsert("d", -0.5) # d becomes root.
        iw_match(self, s.peek(), "d", 0.5)
        s.upsert("f", 9) # f is pushed down to a leaf.
        iw_match(self, s.peek(), "d", 0.5)
        s.upsert("d", -0.5) # root is eliminated.
        iw_match(self, s.peek(), "b", 4)

        s.upsert("d", 1)  # restore it.
        iw_match(self, s.peek(), "d", 1)

        s.upsert("a", -8)  # eliminate a middle node.
        iw_match(self, s.peek(), "d", 1)

        s.upsert("f", -10)  # eliminate a leaf.
        iw_match(self, s.peek(), "d", 1)

        s.upsert("e", -9)  # eliminate terminal leaf.
        iw_match(self, s.peek(), "d", 1)


class TestOnlineHeaviestItems(unittest.TestCase):

    
    def test_works(self):
        s = OnlineHeaviestItems(2)
        # Ignores initially non-positive elements.
        s.upsert("a", 0)
        self.assertEqual(s.heaviest(), [])

        # Updates in unordered state.
        s.upsert("a", 1)
        contents_match(self, s.heaviest(), [("a", 1)])

        # Drops non-positive in unordered state.
        s.upsert("a", -1)
        self.assertEqual(s.heaviest(), [])

        # At capacity.
        s.upsert("a", 1)
        s.upsert("b", 1)
        contents_match(self, s.heaviest(), [("a", 1), ("b", 1)], ordered=False)

        # Update weight at capacity.
        s.upsert("a", 1)
        contents_match(self, s.heaviest(), [("a", 2), ("b", 1)], ordered=False)

        # Add a negligible element.
        s.upsert("c", 0.5)
        contents_match(self, s.heaviest(), [("a", 2), ("b", 1)], ordered=False)
        self.assertEqual(s.bottoms.size(), 1)

        # Push negligible element in the top.
        s.upsert("c", 1)
        contents_match(self, s.heaviest(), [("a", 2), ("c", 1.5)], ordered=False)
        self.assertEqual(s.bottoms.size(), 1) # b is now in bottoms
        iw_match(self, s.bottoms.peek(), "b", 1)

        # Push a top element into the reservoir.
        s.upsert("a", -1.5)
        contents_match(self, s.heaviest(), [("c", 1.5), ("b", 1)], ordered=False)
        self.assertEqual(s.bottoms.size(), 1)
        iw_match(self, s.bottoms.peek(), "a", 0.5)

        # Completely drop an element from the top.
        s.upsert("c", -1.5)
        contents_match(self, s.heaviest(), [("b", 1), ("a", 0.5)], ordered=False)
        self.assertEqual(s.bottoms.size(), 0)

        # Drop again, knowing that reservoir is already empty.
        s.upsert("a", -0.5)
        contents_match(self, s.heaviest(), [("b", 1)], ordered=False)
        s.upsert("b", -1)
        self.assertEqual(s.heaviest(), [])

        # Check that the reservoir is heap-sorted by making sure the heaviest item
        # from the reservoir gets promoted.
        s = OnlineHeaviestItems(2)
        s.upsert("a", 1)
        s.upsert("b", 2)
        s.upsert("c", 3)
        s.upsert("d", 4)
        s.upsert("e", 5)
        contents_match(self, s.heaviest(), [("d", 4), ("e", 5)], ordered=False)
        self.assertEqual(s.bottoms.size(), 3)

        s.upsert("e", -4) # e becomes 1, c (3) from bottoms should be promoted
        contents_match(self, s.heaviest(), [("d", 4), ("c", 3)], ordered=False)

        # Check that elements in the reservoir get their weights correctly updated.
        s.upsert("a", 1)  # positive update.
        s.upsert("b", -1)  # negative update.
        contents_match(self, s.heaviest(), [("d", 4), ("c", 3)], ordered=False)
        # Remove all other elements to show both a and b.
        s.upsert("e", -100)
        s.upsert("d", -100)
        s.upsert("c", -100)
        contents_match(self, s.heaviest(), [("a", 2), ("b", 1)], ordered=False)

    
    def test_limit_1(self):
        s = OnlineHeaviestItems(1)
        s.upsert("a", 5)
        contents_match(self, s.heaviest(), [("a", 5)])
        s.upsert("b", 6)
        contents_match(self, s.heaviest(), [("b", 6)])
        s.upsert("c", 3)
        contents_match(self, s.heaviest(), [("b", 6)])
        s.upsert("b", -2)
        contents_match(self, s.heaviest(), [("a", 5)])


class TestPositiveWeightedItemStoreStress(unittest.TestCase):
    HEAP_SIZES = [127, 128, 129, 1000, 1001]

    def test_online_heap_stress(self):
        for points in self.HEAP_SIZES:
            with self.subTest(points=points):
                min_s = PositiveWeightedItemStore(is_min_heap=True)
                max_s = PositiveWeightedItemStore(is_min_heap=False)
                expected = []
                random.seed(42) # Ensure determinism

                for i in range(points):
                    w = random.random()
                    item = str(i)
                    min_s.upsert(item, w)
                    max_s.upsert(item, w)
                    expected.append(ItemWeight(item, w))

                self.assertEqual(min_s.size(), points)
                self.assertEqual(max_s.size(), points)

                expected.sort(key=lambda x: x.weight)
                for i in range(points):
                    iw = min_s.pop()
                    iw_match(self, iw, expected[i].item, expected[i].weight)
                self.assertEqual(min_s.size(), 0)

                expected.sort(key=lambda x: x.weight, reverse=True)
                for i in range(points):
                    iw = max_s.pop()
                    iw_match(self, iw, expected[i].item, expected[i].weight)
                self.assertEqual(max_s.size(), 0)

    def test_erasure_stress(self):
        # Fill the heap then eliminate ~50% of entries at random using negative
        # weight updates.
        for points in self.HEAP_SIZES:
            with self.subTest(points=points):
                min_s = PositiveWeightedItemStore(is_min_heap=True)
                max_s = PositiveWeightedItemStore(is_min_heap=False)
                kept = []
                to_delete = []
                random.seed(42) # Ensure determinism
                weights = {}

                for i in range(points):
                    w = random.random()
                    item = str(i)
                    weights[item] = w
                    min_s.upsert(item, w)
                    max_s.upsert(item, w)
                    if random.random() < 0.5:
                        kept.append(ItemWeight(item, w))
                    else:
                        to_delete.append(item)

                self.assertEqual(min_s.size(), points)
                self.assertEqual(max_s.size(), points)

                random.shuffle(to_delete)
                for item in to_delete:
                    min_s.upsert(item, -weights[item] - 1) # Ensure weight becomes negative
                    max_s.upsert(item, -weights[item] - 1)

                self.assertEqual(min_s.size(), len(kept))
                self.assertEqual(max_s.size(), len(kept))

                kept.sort(key=lambda x: x.weight)
                for i in range(len(kept)):
                    iw = min_s.pop()
                    iw_match(self, iw, kept[i].item, kept[i].weight)
                self.assertEqual(min_s.size(), 0)

                kept.sort(key=lambda x: x.weight, reverse=True)
                for i in range(len(kept)):
                    iw = max_s.pop()
                    iw_match(self, iw, kept[i].item, kept[i].weight)
                self.assertEqual(max_s.size(), 0)

if __name__ == "__main__":
    unittest.main()
