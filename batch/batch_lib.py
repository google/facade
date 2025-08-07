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
"""Library functions used by batch binaries."""

import datetime
import random
from typing import List

from action import action_source
from common.pipeline_type import PipelineType
from common import time_utils
from context import context_source
from pipelines.merger import pipeline as merger_pipeline
from protos import contextualized_actions_pb2, directive_pb2


def compute_contextualized_actions(
    directive: directive_pb2.Directive,
    pipeline_type: PipelineType,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    action_file_path: str,
    context_file_path: str):
  # Get the snapshot times for the given time range.
  snapshot_period = time_utils.convert_proto_duration_to_timedelta(
      directive.dataset_parameters.snapshot_period)
  if snapshot_period <= datetime.timedelta():
    raise ValueError('Snapshot period must be positive.')
  snapshot_time_offset = time_utils.convert_proto_duration_to_timedelta(
      directive.dataset_parameters.snapshot_time_offset)
  max_num_actions_per_contextualized_actions = directive.dataset_parameters.max_num_actions_per_contextualized_actions
  if max_num_actions_per_contextualized_actions <= 0:
    raise ValueError('Max number of actions per ContextualizedActions must be positive.')
  snapshot_times = time_utils.generate_snapshot_times(
      start_time, end_time, snapshot_period, snapshot_time_offset);
  # Maximum number of contextualized actions per principal and snapshot time at
  # training.
  max_num_cas = 0
  if pipeline_type != PipelineType.INFERENCE:
    max_num_cas = directive.dataset_parameters.max_num_contextualized_actions_per_principal_snapshot

  # Compute featurized actions per action source.
  actions_per_source = dict()
  for action_source_config in directive.action_sources:
    source = action_source.FeaturizationSource(
      action_source_config, pipeline_type, action_file_path)
    actions_by_principal = source.get_actions_by_principal(start_time, end_time)
    action_tuples = [(principal, action)
                     for principal, action_list in actions_by_principal.items()
                     for action in action_list]
    actions_per_source[action_source_config.type] = action_tuples

  # Compute featurized contexts as a single flattened collection.
  contexts_by_source = []
  for context_source_config in directive.context_sources:
    contexts_by_source.extend(
        context_source.get_featurized_context(context_source_config, snapshot_times, context_file_path))

  # Contextualize actions and contexts, and return the result.
  return merger_pipeline.contextualize_actions(
      contexts_by_source,
      actions_per_source,
      snapshot_times,
      max_num_actions_per_contextualized_actions,
      max_num_cas)


def downsample_missing_actions(
    contextualized_actions: List[contextualized_actions_pb2.ContextualizedActions],
    keep_ratio: float
) -> List[contextualized_actions_pb2.ContextualizedActions]:
  """
  Filters a list, randomly dropping ContextualizedActions with empty actions.

  - If an item has a non-empty `actions` list, it is always kept.
  - If an item has an empty `actions` list, it is kept with a
    probability equal to `keep_ratio`.
  """
  if not 0.0 <= keep_ratio <= 1.0:
    raise ValueError("keep_ratio must be between 0.0 and 1.0.")

  filtered_list = []
  for item in contextualized_actions:
    if not item.actions:
      if random.random() < keep_ratio:
        filtered_list.append(item)
    else:
      filtered_list.append(item)
  return filtered_list
