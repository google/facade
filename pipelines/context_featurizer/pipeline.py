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

"""Pipeline for building bipartite features from contexts."""

import collections
import datetime
from typing import Iterable, List, Tuple

from common import time_utils
from pipelines.context_featurizer.pipeline_utils import featurize_bipartite_peer_attributes
from pipelines.context_featurizer.pipeline_utils import reduce_bipartite_peer_attributes
from protos import context_pb2
from protos import context_source_config_pb2
from protos import contextualized_actions_pb2


def build_bipartite_features(
    contexts: Iterable[context_pb2.Context],
    config: context_source_config_pb2.ContextSourceConfig,
    snapshot_times: List[datetime.datetime],
) -> List[Tuple[str, contextualized_actions_pb2.FeaturizedContext]]:
  """Builds bipartite features from contexts based on the provided config."""
  source_type = config.type

  # Reduce the contexts based on snapshot times and peer attributes.
  reduced_contexts: Iterable[context_pb2.Context] = (
      reduce_bipartite_peer_attributes(contexts, config, snapshot_times)
  )

  # Featurize the reduced contexts.
  featurized_data: List[
      Tuple[str, contextualized_actions_pb2.FeaturizedContext]
  ] = featurize_bipartite_peer_attributes(reduced_contexts, config)

  grouped_features = collections.defaultdict(list)
  for principal, context in featurized_data:
    if len(context.features_per_source) != 1:
      raise ValueError(
          "Input FeaturizedContext must have only one ContextSourceFeature but "
          f"it has {len(context.features_per_source)}."
      )
    if len(context.features_per_source[0].features) != 1:
      raise ValueError(
          "Input's ContextSourceFeature must have only one Feature but "
          f"it has {len(context.features_per_source[0].features)}."
      )

    key = (principal, time_utils.convert_proto_time_to_time(context.valid_from))
    grouped_features[key].append(context)

  # Merge features for the same principal and snapshot time.
  merged_features = {}
  for key, contexts_to_merge in grouped_features.items():
    accumulator = contextualized_actions_pb2.ContextSourceFeatures()
    for context in contexts_to_merge:
      if context.features_per_source:
        accumulator.features.extend(context.features_per_source[0].features)
    merged_features[key] = accumulator

  # Prepare the final list of (principal, FeaturizedContext).
  final_results = []
  for key, features in merged_features.items():
    principal, valid_from = key
    result = contextualized_actions_pb2.FeaturizedContext(
        valid_from=time_utils.convert_to_proto_time(valid_from),
        features_per_source=[
            contextualized_actions_pb2.ContextSourceFeatures(
                source_type=source_type, features=features.features
            )
        ],
    )
    final_results.append((principal, result))

  return final_results

