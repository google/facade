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


import heapq
import dataclasses
from typing import List, Tuple, Dict, Optional


@dataclasses.dataclass
class ItemWeight:
    item: str
    weight: float

    def __lt__(self, other: 'ItemWeight') -> bool:
        return self.weight < other.weight

    def __gt__(self, other: 'ItemWeight') -> bool:
        return self.weight > other.weight

    def __eq__(self, other: 'ItemWeight') -> bool:
        return self.weight == other.weight


class PositiveWeightedItemStore:
    """Stores ItemWeights and allows efficient heap operations."""

    def __init__(self, is_min_heap: bool = True):
        self._elements: List[ItemWeight] = []
        self._index: Dict[str, int] = {}
        self._is_min_heap = is_min_heap

    def size(self) -> int:
        return len(self._elements)

    def contents(self) -> List[ItemWeight]:
        return self._elements

    def _get_sign(self) -> int:
        return 1 if self._is_min_heap else -1

    def peek(self) -> ItemWeight:
        if not self._elements:
            raise IndexError("Heap is empty")
        return self._elements[0]

    # If allow_insertion is true, inserts a new item or additively updates the
    # weight of the already-stored item.
    # If allow_insertion is false, only updates the weight of an existing item,
    # and returns false iff the item does not already exist.
    # Regardless of insertion or update, if the stored weight is or becomes
    # non-positive, the item is dropped from the store.
    def upsert(self, item: str, delta: float, allow_insertion: bool = True) -> bool:
        if item in self._index:
            idx = self._index[item]
            existing_weight = self._elements[idx].weight
            new_weight = existing_weight + delta

            if new_weight <= 0:
                self._remove_at(idx)
                return True

            self._elements[idx].weight = new_weight
            if delta > 0:
                self._heapify_down(idx)
            else:
                self._heapify_up(idx)
            return True
        else:
            if not allow_insertion or delta <= 0:
                return False
            self._index[item] = len(self._elements)
            self._elements.append(ItemWeight(item, delta))
            self._heapify_up(len(self._elements) - 1)
            return True

    def _remove_at(self, idx: int):
        item_to_remove = self._elements[idx].item
        del self._index[item_to_remove]

        if idx == len(self._elements) - 1:
            self._elements.pop()
            return

        self._elements[idx] = self._elements.pop()
        self._index[self._elements[idx].item] = idx
        self._heapify_down(idx)
        self._heapify_up(idx) # In case the swapped element needs to go up

    def pop(self) -> ItemWeight:
        if not self._elements:
            raise IndexError("Heap is empty")

        result = self._elements[0]
        del self._index[result.item]

        if len(self._elements) == 1:
            self._elements.pop()
            return result

        self._elements[0] = self._elements.pop()
        self._index[self._elements[0].item] = 0
        self._heapify_down(0)
        return result

    def heapify(self):
        # Python's heapq is always a min-heap, so we adjust weights for max-heap
        if not self._is_min_heap:
            # This conversion is not directly supported in the Python version
            # as heapq operations are done in place.
            # Instead, the min/max heap property is handled by weight signs.
            pass

    def _parent(self, i: int) -> int:
        return (i - 1) // 2

    def _left_child(self, i: int) -> int:
        return 2 * i + 1

    def _right_child(self, i: int) -> int:
        return 2 * i + 2

    def _swap(self, i: int, j: int):
        self._index[self._elements[i].item] = j
        self._index[self._elements[j].item] = i
        self._elements[i], self._elements[j] = self._elements[j], self._elements[i]

    def _compare(self, w1: float, w2: float) -> bool:
        """Returns True if w1 should be higher priority than w2."""
        if self._is_min_heap:
            return w1 < w2
        return w1 > w2

    def _heapify_up(self, i: int):
        while i > 0 and self._compare(self._elements[i].weight, self._elements[self._parent(i)].weight):
            self._swap(i, self._parent(i))
            i = self._parent(i)

    def _heapify_down(self, i: int):
        size = len(self._elements)
        while True:
            l = self._left_child(i)
            r = self._right_child(i)
            smallest_or_largest = i

            if l < size and self._compare(self._elements[l].weight, self._elements[smallest_or_largest].weight):
                smallest_or_largest = l

            if r < size and self._compare(self._elements[r].weight, self._elements[smallest_or_largest].weight):
                smallest_or_largest = r

            if smallest_or_largest != i:
                self._swap(i, smallest_or_largest)
                i = smallest_or_largest
            else:
                break

class OnlineHeaviestItems:
    """Keeps track of the top-n heaviest string-identified items in a stream."""

    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.tops = PositiveWeightedItemStore(is_min_heap=True)  # Min-heap for the top N items
        self.bottoms = PositiveWeightedItemStore(is_min_heap=False)  # Max-heap for the rest

    def upsert(self, item: str, weight: float):
        # Try to update in tops first
        in_tops = item in self.tops._index
        if in_tops:
            self.tops.upsert(item, weight)
        else:
            # Try to update in bottoms
            in_bottoms = item in self.bottoms._index
            if in_bottoms:
                self.bottoms.upsert(item, weight)
            else:
                # Item is new
                if self.tops.size() < self.limit:
                    self.tops.upsert(item, weight)
                elif weight > self.tops.peek().weight:
                    self.tops.upsert(item, weight)
                elif weight > 0:
                    self.bottoms.upsert(item, weight)

        # Maintain invariants
        self._balance_heaps()

    def _balance_heaps(self):
        # If tops has space, move from bottoms
        while self.tops.size() < self.limit and self.bottoms.size() > 0:
            iw = self.bottoms.pop()
            self.tops.upsert(iw.item, iw.weight)

        # If tops is over capacity, move to bottoms
        while self.tops.size() > self.limit:
            iw = self.tops.pop()
            self.bottoms.upsert(iw.item, iw.weight)

        # Maintain min(tops) >= max(bottoms)
        while (
            self.tops.size() > 0
            and self.bottoms.size() > 0
            and self.bottoms.peek().weight > self.tops.peek().weight
        ):
            iw_high = self.bottoms.pop()
            iw_low = self.tops.pop()
            self.tops.upsert(iw_high.item, iw_high.weight)
            self.bottoms.upsert(iw_low.item, iw_low.weight)

    def heaviest(self) -> List[ItemWeight]:
        return self.tops.contents()
