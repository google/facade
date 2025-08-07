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
import inspect
import math

import tensorflow as tf

from model.metrics import concatenator


class ConcatenatorTest(tf.test.TestCase):

  def test_next_array_size(self):
    # Tests that next_array_size() always returns a large-enough array size.
    # Test around the boundaries where off-by-one errors are most likely.
    validity_limit = 2**50
    exponent_limit = math.ceil(
        math.log(validity_limit) / math.log(concatenator._GROWTH_FACTOR)
    )
    for exponent in range(exponent_limit):
      center = math.ceil(concatenator._GROWTH_FACTOR**exponent)
      for x in range(max(1, center - 100), center + 100):
        self.assertGreaterEqual(concatenator._next_array_size(x), x)

  def test_concatenates(self):
    conc = concatenator.Concatenator(dtype=tf.int32)
    expected = []
    self.assertAllEqual(conc.result(), expected)
    conc.update_state([])  # Handles empty update.
    self.assertAllEqual(conc.result(), expected)
    for i in range(100):
      expected.append(i)
      conc.update_state([i])
      self.assertAllEqual(conc.result(), expected)

  def test_resets(self):
    conc = concatenator.Concatenator(dtype=tf.int32)
    conc.update_state([0, 1, 2])
    self.assertAllEqual(conc.result(), [0, 1, 2])
    conc.reset_state()
    self.assertAllEqual(conc.result(), [])
    conc.update_state([3, 4])
    self.assertAllEqual(conc.result(), [3, 4])

  def test_merges_states(self):
    conc1 = concatenator.Concatenator(dtype=tf.int32)
    conc2 = concatenator.Concatenator(dtype=tf.int32)
    conc1.update_state([0, 1, 2])
    conc2.update_state([3, 4, 5, 6])
    conc1.merge_state([conc2])
    self.assertAllEqual(conc1.result(), [0, 1, 2, 3, 4, 5, 6])

  def test_serializes(self):
    config = {"dtype": tf.float64}
    f = concatenator.Concatenator.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(concatenator.Concatenator).parameters.keys(),
        config.keys(),
    )


if __name__ == "__main__":
  tf.test.main()
