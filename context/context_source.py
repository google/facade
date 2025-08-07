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

"""Utilities for handling Context data and configurations."""

import datetime
from typing import List, Tuple

from common import source_data
from common import time_utils
from pipelines.context_featurizer.pipeline import build_bipartite_features
from protos import context_source_config_pb2
from protos import contextualized_actions_pb2

ResultsByPrincipal = List[
    Tuple[str, contextualized_actions_pb2.FeaturizedContext]
]


def get_featurized_context(
    config: context_source_config_pb2.ContextSourceConfig,
    snapshot_times: List[datetime.datetime],
    tf_record_file: str,
) -> ResultsByPrincipal:
  """Gets featurized contexts from the source.

  Args:
    config: The context source configuration.
    snapshot_times: A list of snapshot times.
    tf_record_file: The path to the TFRecord file containing Context protos.

  Returns:
    A dictionary mapping principal to FeaturizedContext.

  Raises:
    ValueError: If snapshot_times is empty.
  """
  if not snapshot_times:
    raise ValueError("snapshot_times must not be empty.")

  start_time = snapshot_times[
      0
  ] - time_utils.convert_proto_duration_to_timedelta(config.context_lookback)
  end_time = snapshot_times[-1]

  all_contexts = source_data.read_context(
      config.type, tf_record_file, start_time, end_time
  )

  bipartite_features: ResultsByPrincipal = build_bipartite_features(
      all_contexts, config, snapshot_times
  )

  return bipartite_features

