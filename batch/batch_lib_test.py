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

from batch import batch_lib
from protos import contextualized_actions_pb2


class TestDownSample(unittest.TestCase):

  def setUp(self):
    self.test_data = []
    for i in range(3):
      actions = [contextualized_actions_pb2.FeaturizedActionsBySource(source_type="action_source")] if i % 2 == 0 else []
      self.test_data.append(
          contextualized_actions_pb2.ContextualizedActions(actions=actions, principal=f"user{i}")
      )

      
  def test_works_keeps_all(self):
      result = batch_lib.downsample_missing_actions(self.test_data, 1.0)
      self.assertCountEqual(result, self.test_data)


  def test_works_drops_actionless(self):
      result = batch_lib.downsample_missing_actions(self.test_data, 0.0)
      expected_result = [self.test_data[0], self.test_data[2]]
      self.assertCountEqual(result, expected_result)


if __name__ == '__main__':
  unittest.main()
