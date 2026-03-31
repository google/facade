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
"""Tests for string_pool module."""

import unittest
import sys

from common import string_pool


class StringPoolTest(unittest.TestCase):

  def setUp(self):
    """Create a fresh string pool for each test."""
    self.pool = string_pool.StringPool()

  def test_basic_interning(self):
    """Test that identical strings return the same object."""
    s1 = "test_principal"
    s2 = "test_principal"
    
    interned1 = self.pool.intern_principal(s1)
    interned2 = self.pool.intern_principal(s2)
    
    # Should be the exact same object in memory
    self.assertIs(interned1, interned2)
    
  def test_separate_pools(self):
    """Test that different pool types are separate."""
    s = "shared_name"
    
    principal = self.pool.intern_principal(s)
    resource = self.pool.intern_resource(s)
    action_id = self.pool.intern_action_id(s)
    
    # All should be interned, but in separate pools
    stats = self.pool.get_stats()
    self.assertEqual(stats['principals'], 1)
    self.assertEqual(stats['resources'], 1)
    self.assertEqual(stats['action_ids'], 1)

  def test_bytes_interning(self):
    """Test that bytes are interned."""
    b1 = b"test_bytes"
    b2 = b"test_bytes"
    
    interned1 = self.pool.intern_bytes(b1)
    interned2 = self.pool.intern_bytes(b2)
    
    self.assertIs(interned1, interned2)

  def test_decode_and_intern(self):
    """Test combined decode and intern operation."""
    b = b"principal_name"
    
    s1 = self.pool.decode_and_intern(b, 'principal')
    s2 = self.pool.decode_and_intern(b, 'principal')
    
    self.assertIs(s1, s2)
    self.assertEqual(s1, "principal_name")
    
    stats = self.pool.get_stats()
    self.assertEqual(stats['principals'], 1)

  def test_encode_and_intern(self):
    """Test combined intern and encode operation."""
    s = "principal_name"
    
    b1 = self.pool.encode_and_intern(s, 'principal')
    b2 = self.pool.encode_and_intern(s, 'principal')
    
    # Both the string and bytes should be interned
    self.assertIs(b1, b2)
    self.assertEqual(b1, b"principal_name")

  def test_memory_savings(self):
    """Test that interning saves memory with many duplicates."""
    # Create many copies of the same principal name
    principal_name = "alice@example.com"
    
    # Without interning, each would be a separate object
    duplicates = [principal_name for _ in range(10000)]
    
    # With interning, all point to the same object
    interned = [self.pool.intern_principal(s) for s in duplicates]
    
    # Verify all are the same object
    for i in range(len(interned) - 1):
      self.assertIs(interned[i], interned[i + 1])
    
    # Pool should only have 1 unique string
    stats = self.pool.get_stats()
    self.assertEqual(stats['principals'], 1)

  def test_stats(self):
    """Test statistics tracking."""
    self.pool.intern_principal("alice")
    self.pool.intern_principal("bob")
    self.pool.intern_principal("alice")  # duplicate
    
    self.pool.intern_resource("doc1")
    self.pool.intern_resource("doc2")
    
    self.pool.intern_action_id("action1")
    
    stats = self.pool.get_stats()
    
    self.assertEqual(stats['principals'], 2)  # alice, bob
    self.assertEqual(stats['resources'], 2)   # doc1, doc2
    self.assertEqual(stats['action_ids'], 1)  # action1
    self.assertEqual(stats['total_strings'], 5)

  def test_global_pool(self):
    """Test that global pool functions work."""
    # Clear global pool first
    string_pool.get_global_pool().clear()
    
    s1 = string_pool.intern_principal("global_test")
    s2 = string_pool.intern_principal("global_test")
    
    self.assertIs(s1, s2)
    
    stats = string_pool.get_pool_stats()
    self.assertEqual(stats['principals'], 1)

  def test_clear(self):
    """Test that clear removes all entries."""
    self.pool.intern_principal("alice")
    self.pool.intern_resource("doc1")
    
    stats = self.pool.get_stats()
    self.assertGreater(stats['total_strings'], 0)
    
    self.pool.clear()
    
    stats = self.pool.get_stats()
    self.assertEqual(stats['total_strings'], 0)

  def test_actual_memory_savings(self):
    """Demonstrate actual memory savings."""
    if sys.version_info < (3, 9):
      self.skipTest("sys.getsizeof behavior varies in Python < 3.9")
    
    # Create a realistic scenario
    principal_name = "very_long_principal_name_that_appears_frequently@company.com"
    num_occurrences = 1000
    
    # Measure size without interning
    without_interning = [principal_name for _ in range(num_occurrences)]
    size_without = sum(sys.getsizeof(s) for s in without_interning)
    
    # Measure size with interning (all point to same object)
    with_interning = [self.pool.intern_principal(principal_name) 
                      for _ in range(num_occurrences)]
    # Size is just the one string + references
    size_with = sys.getsizeof(principal_name) + sys.getsizeof(with_interning)
    
    # Should save significant memory
    savings_ratio = size_without / size_with
    print(f"\nMemory savings ratio: {savings_ratio:.1f}x")
    print(f"Without interning: {size_without:,} bytes")
    print(f"With interning: {size_with:,} bytes")
    
    # Should save at least 10x with 1000 duplicates
    self.assertGreater(savings_ratio, 10)


if __name__ == '__main__':
  unittest.main()
