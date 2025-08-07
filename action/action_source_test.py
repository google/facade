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


import datetime
import os
import tempfile
from typing import Dict, List, Tuple
import unittest
# Import the module to test
from action import action_source
from common.pipeline_type import PipelineType
from pipelines.history_featurizer import pipeline
from protos import action_pb2, action_source_config_pb2, contextualized_actions_pb2
import tensorflow as tf

Feature = contextualized_actions_pb2.Feature
FeaturizedAction = contextualized_actions_pb2.FeaturizedAction
Action = action_pb2.Action
ActionSourceConfig = action_source_config_pb2.ActionSourceConfig
WeightedToken = contextualized_actions_pb2.WeightedToken


def time_by_hour(hour: int) -> datetime.datetime:
  return datetime.datetime.fromtimestamp(hour * 3600, tz=datetime.timezone.utc)


def make_action(
    action_id: bytes,
    occurred_at: datetime.datetime,
    principal: str,
    resource_id: str,
) -> Action:
  a = Action(
      id=str(action_id).encode("utf-8"),
      type="drive",
      principal=principal,
      resource_id=resource_id,
  )
  a.occurred_at.seconds = int(occurred_at.timestamp())
  a.history_key = resource_id.encode("utf-8")
  return a


def make_history_featurized_action(
    action_id: str,
    resource_id: str,
    occurred_at: datetime.datetime,
    token: str = "",
    weight: int = 0,
) -> FeaturizedAction:
  fa = FeaturizedAction(
      id=action_id.encode("utf-8"),
      resource_id=resource_id,
  )
  fa.occurred_at.seconds = int(occurred_at.timestamp())
  fa.occurred_at.nanos = int(
      (occurred_at.timestamp() - fa.occurred_at.seconds) * 1e9
  )

  f = fa.features.add(name="prin")
  bow = f.bag_of_weighted_words
  if token:
    wt = bow.tokens.add(token=token.encode("utf-8"), weight=weight)
  return fa


def write_tfrecord(file_path: str, actions: List[Action]):
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with tf.io.TFRecordWriter(file_path) as writer:
    for action in actions:
      writer.write(action.SerializeToString())


class ActionSourceTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir.name

  def assert_dict_equal(self, d1, d2, msg=None):
    self.assertEqual(len(d1), len(d2), msg)
    for k, v in d1.items():
      self.assertIn(k, d2, msg)
      self.assertEqual(v, d2[k], msg)

  def test_history_features(self):
    u1 = "user1"
    u2 = "user2"
    d1 = "doc1"
    d2 = "doc2"

    raw_actions = [
        make_action("1", time_by_hour(2), u1, d1),
        make_action("3", time_by_hour(3), u1, d1),
        make_action("2", time_by_hour(3), u2, d2),
    ]
    tfrecord_path = os.path.join(self.test_dir, "drive.tfrecord")
    write_tfrecord(tfrecord_path, raw_actions)

    config = ActionSourceConfig(
        type="drive",
    )
    config.history_duration.seconds = 7200
    config.action_deduplication_period.seconds = 1
    config.max_history_keys_per_day = 100
    config.max_values_per_feature = 100

    source = action_source.FeaturizationSource(
        config, PipelineType.TRAINING, tfrecord_path
    )
    results = source.get_actions_by_principal(time_by_hour(1), time_by_hour(4))

    expected_fa1 = make_history_featurized_action(
        "1", d1, time_by_hour(2), "", 0
    )
    expected_fa2 = make_history_featurized_action(
        "2", d2, time_by_hour(3), "", 0
    )
    expected_fa3 = make_history_featurized_action(
        "3", d1, time_by_hour(3), u1, 1.0
    )

    expected = {
        u1: [expected_fa1, expected_fa3],
        u2: [expected_fa2],
    }

    self.assert_dict_equal(results, expected)

  def test_filters_history_with_too_few_tokens(self):
    u1 = "user1"
    u2 = "user2"
    d1 = "doc1"

    raw_actions = [
        make_action("1", time_by_hour(2), u1, d1),
        make_action("2", time_by_hour(3), u2, d1),
    ]

    tfrecord_path = os.path.join(self.test_dir, "drive.tfrecord")
    write_tfrecord(tfrecord_path, raw_actions)

    config = ActionSourceConfig(
        type="drive",
    )
    config.history_duration.seconds = 7200
    config.action_deduplication_period.seconds = 1
    config.max_history_keys_per_day = 100
    config.max_values_per_feature = 100
    config.min_values_per_feature = 1

    source = action_source.FeaturizationSource(
        config, PipelineType.TRAINING, tfrecord_path
    )
    results = source.get_actions_by_principal(time_by_hour(1), time_by_hour(4))

    # Action "1" (u1, d1) at TimeByHour(2) would produce 0 tokens. Since
    # min_values_per_feature is 1, it's dropped.
    # Action "2" (u2, d1) at TimeByHour(3) has Action "1" in history.
    # Its history feature for "prin" will include u1.

    expected_fa2 = make_history_featurized_action(
        "2", d1, time_by_hour(3), u1, 1.0
    )

    # The expected dictionary structure for the Python test
    expected = {
        u2: [expected_fa2],
    }

    self.assert_dict_equal(results, expected)

  def test_inference_pipeline_ignore_min_values_per_feature(self):
    u1 = "user1"
    u2 = "user2"
    d1 = "doc1"

    raw_actions = [
        make_action("1", time_by_hour(2), u1, d1),
        make_action("2", time_by_hour(3), u2, d1),
    ]
    tfrecord_path = os.path.join(self.test_dir, "drive.tfrecord")
    write_tfrecord(tfrecord_path, raw_actions)

    config = ActionSourceConfig(
        type="drive",
    )

    config.history_duration.seconds = 7200
    config.action_deduplication_period.seconds = 1
    config.max_history_keys_per_day = 100
    config.max_values_per_feature = 100
    config.min_values_per_feature = 1

    source = action_source.FeaturizationSource(
        config, PipelineType.INFERENCE, tfrecord_path
    )
    results = source.get_actions_by_principal(time_by_hour(1), time_by_hour(4))

    # BOTH actions are kept, even though action1 has no features, because it's INFERENCE mode.
    expected = {
        u1: [make_history_featurized_action("1", d1, time_by_hour(2), "", 0)],
        u2: [make_history_featurized_action("2", d1, time_by_hour(3), u1, 1)],
    }

    self.assert_dict_equal(results, expected)


if __name__ == "__main__":
  unittest.main()
