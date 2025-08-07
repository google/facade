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

"""Tests for history featurizer pipeline."""

import datetime
import math
import unittest
from typing import Dict, List

from protos import timestamp_pb2, contextualized_actions_pb2, action_pb2
from pipelines.history_featurizer import pipeline

Action = action_pb2.Action
Feature = contextualized_actions_pb2.Feature
WeightedToken = contextualized_actions_pb2.WeightedToken


def make_action(
    action_id: bytes,
    time: datetime.datetime,
    key: str,
    principal: str,
) -> Action:
  
  a = Action(id=str(action_id).encode("utf-8"), principal=principal)
  a.occurred_at.seconds = int(time.timestamp())
  a.history_key = key.encode("utf-8")
  # print(a)
  return a


def make_weighted_token(token: str, weight: float) -> WeightedToken:
  return WeightedToken(token=token.encode("utf-8"), weight=weight)


def make_users_feature(tokens: list[WeightedToken]) -> Feature:
  f = Feature(name="prin")
  # Sort tokens by token name for consistent comparison
  tokens.sort(key=lambda t: t.token)
  f.bag_of_weighted_words.tokens.extend(tokens)
  return f


def compare_weighted_tokens(a: WeightedToken, b: WeightedToken, places=5) -> bool:
  return a.token == b.token and math.isclose(a.weight, b.weight, rel_tol=1e-9, abs_tol=10**(-places))


def compare_feature(a: Feature, b: Feature, places=5) -> bool:
  if a.name != b.name:
    return False
  if len(a.bag_of_weighted_words.tokens) != len(b.bag_of_weighted_words.tokens):
    return False

  a_tokens = sorted(a.bag_of_weighted_words.tokens, key=lambda t: t.token)
  b_tokens = sorted(b.bag_of_weighted_words.tokens, key=lambda t: t.token)

  for token_a, token_b in zip(a_tokens, b_tokens):
    if not compare_weighted_tokens(token_a, token_b, places):
      return False
  return True


class BuildHistoryFeaturesTest(unittest.TestCase):

  def assertResultDictEqual(self, expected: Dict[str, List[Feature]], actual: Dict[str, List[Feature]], places=5):
    self.assertEqual(set(expected.keys()), set(actual.keys()), msg="Keys differ")

    for key, expected_features in expected.items():
      actual_features = actual[key]
      self.assertEqual(
          len(expected_features), len(actual_features), msg=f"Feature list length diff on key {key}"
      )

      for i, (exp_f, act_f) in enumerate(zip(expected_features, actual_features)):
        self.assertTrue(
            compare_feature(exp_f, act_f, places),
            msg=f"Features differ on key {key} at index {i}:\nExpected: {exp_f}\nActual: {act_f}",
        )
  

  def test_respects_start_and_end_times_for_same_target(self):
    t0 = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    start_time = t0 + datetime.timedelta(hours=24)
    end_time = start_time + datetime.timedelta(hours=25)
    target = "target"
    target2 = "target2"

    actions = [
        # First four actions are before start_time, they should never appear in
        # the results.
        make_action(0, t0, target, ""),
        make_action(1, t0, target, "value"),
        make_action(2, start_time - datetime.timedelta(seconds=1), target, ""),
        make_action(3, start_time - datetime.timedelta(seconds=1), target, "value"),
        # These are at or after start_time, they should appear regardless of
        # whether they have history.
        make_action(4, start_time, target, ""),
        make_action(5, start_time, target, "value"),
        make_action(6, start_time + datetime.timedelta(hours=24), target, ""),
        make_action(7, start_time + datetime.timedelta(hours=24), target, "value"),
        make_action(8, start_time, target2, ""),
        # Finally, these are at or after end_time, they should never appear.
        make_action(9, start_time + datetime.timedelta(hours=25), target, "value"),
        make_action(10, start_time + datetime.timedelta(hours=25), target2, "value"),
    ]

    history_duration = datetime.timedelta(0)
    deduplication_period = datetime.timedelta(0)
    max_tokens = 100
    res = pipeline.build_history_features(
        actions,
        start_time,
        end_time,
        history_duration,
        deduplication_period,
        max_tokens,
        max_history_keys_per_day=-1,
    )

    expected_ids = {"4", "5", "6", "7", "8"}
    self.assertEqual(expected_ids, set(res.keys()))

  
  def test_accumulates_intensities(self):
    t0 = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    target = "target"
    actions = [
        make_action(0, t0, target, "value0"),
        make_action(1, t0 + datetime.timedelta(seconds=1), target, "value1"),
        make_action(2, t0 + datetime.timedelta(seconds=2), target, "value0"),
        make_action(3, t0 + datetime.timedelta(seconds=3), target, "value0"),
        make_action(4, t0 + datetime.timedelta(seconds=4), target, "value0"),
    ]
    history_duration = datetime.timedelta(hours=1)
    deduplication_period = datetime.timedelta(0)
    start_time = t0
    end_time = t0 + datetime.timedelta(days=365)
    max_tokens = 100
    res = pipeline.build_history_features(
        actions, start_time, end_time, history_duration, deduplication_period, max_tokens
    )
    wt0 = make_weighted_token("value0", 1.0)
    wt1 = make_weighted_token("value1", 1.0)
    expected = {
        "0": [make_users_feature([])],
        "1": [make_users_feature([wt0])],
        "2": [make_users_feature([wt0, wt1])],
        "3": [make_users_feature([make_weighted_token("value0", 2.0), wt1])],
        "4": [make_users_feature([make_weighted_token("value0", 3.0), wt1])],
    }

    self.assertResultDictEqual(expected, res)
  

  def test_retires_intensities(self):
    t0 = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    target = "target"
    actions = [
        make_action(0, t0, target, "value0"),
        make_action(1, t0 + datetime.timedelta(seconds=1), target, "value0"),
        make_action(2, t0 + datetime.timedelta(seconds=2), target, "value1"),
        make_action(3, t0 + datetime.timedelta(seconds=3), target, "value1"),
        make_action(4, t0 + datetime.timedelta(seconds=4), target, "value1"),
    ]
    history_duration = datetime.timedelta(seconds=2)
    deduplication_period = datetime.timedelta(0)
    start_time = t0
    end_time = t0 + datetime.timedelta(days=365)
    max_tokens = 100
    res = pipeline.build_history_features(
        actions, start_time, end_time, history_duration, deduplication_period, max_tokens
    )
    wt0 = make_weighted_token("value0", 1.0)
    wt1 = make_weighted_token("value1", 1.0)
    wt0_2 = make_weighted_token("value0", 2.0)
    wt1_2 = make_weighted_token("value1", 2.0)
    expected = {
        "0": [make_users_feature([])],
        "1": [make_users_feature([wt0])],
        "2": [make_users_feature([wt0_2])],
        "3": [make_users_feature([wt0, wt1])],
        "4": [make_users_feature([wt1_2])],
    }
    self.assertResultDictEqual(expected, res)
  

  def test_deduplicates_atoms(self):
    t0 = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    target = "target"
    actions = [
        make_action(0, t0, target, "value0"),
        make_action(1, t0 + datetime.timedelta(seconds=1), target, "value0"),  # Not counted
        make_action(2, t0 + datetime.timedelta(seconds=2), target, "value1"),
        make_action(3, t0 + datetime.timedelta(seconds=3), target, "value1"),  # Not counted
        make_action(4, t0 + datetime.timedelta(seconds=4), target, "value1"),
    ]
    history_duration = datetime.timedelta(hours=1)
    deduplication_period = datetime.timedelta(seconds=1)
    start_time = t0
    end_time = t0 + datetime.timedelta(days=365)
    max_tokens = 100
    res = pipeline.build_history_features(
        actions, start_time, end_time, history_duration, deduplication_period, max_tokens
    )
    wt0 = make_weighted_token("value0", 1.0)
    wt1 = make_weighted_token("value1", 1.0)
    expected = {
        "0": [make_users_feature([])],
        "1": [make_users_feature([wt0])],
        "2": [make_users_feature([wt0])],
        "3": [make_users_feature([wt0, wt1])],
        "4": [make_users_feature([wt0, wt1])],
    }

    self.assertResultDictEqual(expected, res)
  

  def test_correct_histories_for_multiple_targets(self):
    t0 = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    target0 = "target0"
    target1 = "target1"
    actions = [
        make_action(0, t0, target0, "value0"),
        make_action(1, t0, target1, "value1"),
        make_action(2, t0 + datetime.timedelta(seconds=1), target0, "value1"),
        make_action(3, t0 + datetime.timedelta(seconds=1), target1, "value0"),
        make_action(4, t0 + datetime.timedelta(seconds=2), target0, "value0"),
    ]
    history_duration = datetime.timedelta(seconds=2)
    deduplication_period = datetime.timedelta(seconds=1)
    start_time = t0
    end_time = t0 + datetime.timedelta(days=365)
    max_tokens = 100
    res = pipeline.build_history_features(
        actions, start_time, end_time, history_duration, deduplication_period, max_tokens
    )
    wt0 = make_weighted_token("value0", 1.0)
    wt1 = make_weighted_token("value1", 1.0)
    expected = {
        "0": [make_users_feature([])],
        "1": [make_users_feature([])],
        "2": [make_users_feature([wt0])],
        "3": [make_users_feature([wt1])],
        "4": [make_users_feature([wt0, wt1])],
    }
    self.assertResultDictEqual(expected, res)
  

  def test_remove_too_frequent_target(self):
    t0 = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    target0, target1, target2, target3 = "t0", "t1", "t2", "t3"
    actions = [
        make_action(0, t0, target0, "v0"),  # 1
        make_action(1, t0 + datetime.timedelta(seconds=1), target1, "v0"),  # 2
        make_action(2, t0 + datetime.timedelta(seconds=2), target1, "v1"),
        make_action(3, t0 + datetime.timedelta(seconds=3), target2, "v0"),  # 3
        make_action(4, t0 + datetime.timedelta(seconds=4), target2, "v1"),
        make_action(5, t0 + datetime.timedelta(seconds=5), target2, "v2"),
        make_action(6, t0 + datetime.timedelta(seconds=6), target3, "v0"),  # 4
        make_action(7, t0 + datetime.timedelta(seconds=7), target3, "v1"),
        make_action(8, t0 + datetime.timedelta(seconds=8), target3, "v2"),
        make_action(9, t0 + datetime.timedelta(seconds=9), target3, "v3"),
    ]
    history_duration = datetime.timedelta(hours=1)
    deduplication_period = datetime.timedelta(0)
    start_time = t0
    end_time = t0 + datetime.timedelta(hours=23)
    max_tokens = 100
    res = pipeline.build_history_features(
        actions,
        start_time,
        end_time,
        history_duration,
        deduplication_period,
        max_tokens,
        max_history_keys_per_day=3,
    )
    wt0 = make_weighted_token("v0", 1.0)
    wt1 = make_weighted_token("v1", 1.0)
    expected = {
        "0": [make_users_feature([])],
        "1": [make_users_feature([])],
        "2": [make_users_feature([wt0])],
        "3": [make_users_feature([])],
        "4": [make_users_feature([wt0])],
        "5": [make_users_feature([wt0, wt1])],
    }
    self.assertResultDictEqual(expected, res)
  

  def test_respects_min_tokens_per_segment(self):
    t0 = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    target0, target1 = "t0", "t1"
    actions = [
        make_action(0, t0, target0, "value0"),
        make_action(1, t0, target1, "value1"),
        make_action(2, t0 + datetime.timedelta(seconds=1), target0, "value1"),
        make_action(3, t0 + datetime.timedelta(seconds=1), target1, "value0"),
        make_action(4, t0 + datetime.timedelta(seconds=2), target0, "value0"),
    ]
    history_duration = datetime.timedelta(seconds=2)
    deduplication_period = datetime.timedelta(seconds=1)
    start_time = t0
    end_time = t0 + datetime.timedelta(days=365)
    max_tokens = 100
    res = pipeline.build_history_features(
        actions,
        start_time,
        end_time,
        history_duration,
        deduplication_period,
        max_tokens,
        min_tokens_per_segment=1,
    )
    wt0 = make_weighted_token("value0", 1.0)
    wt1 = make_weighted_token("value1", 1.0)
    expected = {
        "2": [make_users_feature([wt0])],
        "3": [make_users_feature([wt1])],
        "4": [make_users_feature([wt0, wt1])],
    }

    self.assertResultDictEqual(expected, res)


if __name__ == "__main__":
  unittest.main()
