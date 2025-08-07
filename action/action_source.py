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
import datetime
from datetime import timedelta
from typing import Dict, List, Tuple

from common import time_utils
from common.pipeline_type import PipelineType
from common.source_data import read_actions
from pipelines.history_featurizer.pipeline import build_history_features
from protos import action_pb2, action_source_config_pb2, contextualized_actions_pb2

Feature = contextualized_actions_pb2.Feature
FeaturizedAction = contextualized_actions_pb2.FeaturizedAction
Action = action_pb2.Action
ActionSourceConfig = action_source_config_pb2.ActionSourceConfig


class FeaturizationSource:
  """Reads raw actions, filters them, and generates features."""

  def __init__(
      self,
      config: ActionSourceConfig,
      pipeline_type: PipelineType,
      tfrecord_path: str,
  ):
    self.config = config
    self.pipeline_type = pipeline_type
    self.tfrecord_path = tfrecord_path
    self.config_type = config.type
    self.action_id_to_action = {}

  def _make_featurized_action(
      self, action_id: str, features: List[Feature]
  ) -> FeaturizedAction | None:
    """Creates a FeaturizedAction proto from an Action and its features."""

    # Featureless actions are dropped for training/validation.
    if self.pipeline_type != PipelineType.INFERENCE and not features:
      return None

    result = FeaturizedAction(
        id=action_id.encode("utf-8"),
        resource_id=self.action_id_to_action[action_id].resource_id,
        occurred_at=self.action_id_to_action[action_id].occurred_at,
    )

    if features:
      result.features.extend(features)
    return result

  def get_actions_by_principal(self, start_time, end_time):
    """Creates an action source pipeline for featurizing actions.

    This version is single-process and does not use Flume.
    """
    config_type = self.config.type

    history_duration = time_utils.convert_proto_duration_to_timedelta(self.config.history_duration)
    earliest_event = start_time - history_duration
    raw_actions = read_actions(
        config_type, self.tfrecord_path, earliest_event, end_time
    )

    for a in raw_actions:
      self.action_id_to_action[a.id.decode("utf-8", "ignore")] = a

    # Simplified resource counting
    unique_resources = set(a.resource_id for a in raw_actions)

    history_duration = (
        self.config.history_duration
    )  # history_duration is of type Duration proto
    deduplication_period = (
        self.config.action_deduplication_period
    )  # type is again Duration proto
    max_history_keys_per_day = self.config.max_history_keys_per_day
    max_values_per_feature = self.config.max_values_per_feature
    min_values_per_feature = 0
    if self.pipeline_type != PipelineType.INFERENCE:
      min_values_per_feature = self.config.min_values_per_feature

    # Use all raw_actions for potential history
    actions_with_features = build_history_features(
        actions=raw_actions,
        featurize_actions_at_or_after=start_time,
        featurize_actions_before=end_time,
        history_duration=time_utils.convert_proto_duration_to_timedelta(
            history_duration
        ),
        action_deduplication_period=time_utils.convert_proto_duration_to_timedelta(
            deduplication_period
        ),
        max_tokens_per_segment=max_values_per_feature,
        max_history_keys_per_day=max_history_keys_per_day,
        min_tokens_per_segment=min_values_per_feature,
    )

    results_by_principal = defaultdict(list)
    for action_id, features in actions_with_features.items():
      featurized_action = self._make_featurized_action(action_id, features)
      if featurized_action:
        results_by_principal[
            self.action_id_to_action[action_id].principal
        ].append(featurized_action)

    return results_by_principal
