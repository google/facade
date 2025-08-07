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
"""Used for merging FeaturizedActions and FeaturizedContexts by snapshot times."""

import bisect
import collections
import datetime
from typing import List, Dict, Tuple, Optional, Any

from common import time_utils
from protos import contextualized_actions_pb2


def contextualize_actions(
    contexts: List[Tuple[str, contextualized_actions_pb2.FeaturizedContext]],
    actions: Dict[str, List[Tuple[str, contextualized_actions_pb2.FeaturizedAction]]],
    snapshot_times: List[datetime.datetime],
    max_num_actions_per_ca: int,
    max_num_cas_per_principal_snapshot: int = 0
) -> List[contextualized_actions_pb2.ContextualizedActions]:
  """
  Create ContextualizedActions by merging FeaturizedContexts and FeaturizedActions.

  - If there are more actions than `max_num_actions_per_ca`, the context is
    duplicated to contextualize the excess actions in subsequent
    `ContextualizedActions` objects.
  - If `max_num_cas_per_principal_snapshot` is positive, it caps the number
    of `ContextualizedActions` objects produced for any single principal at
    a given snapshot time.
  - Actions without a corresponding context are dropped.
  - Contexts without any corresponding actions are returned as-is.
  - The actions within each output object are sorted alphabetically by their
    source type.

  Args:
      contexts: A list of tuples, where each tuple contains a principal (str)
                and their FeaturizedContext. It's assumed contexts are aligned
                with snapshot_times.
      actions: A dictionary mapping an action source name (str) to a list of
               tuples, where each tuple is a principal (str) and their
               FeaturizedAction.
      snapshot_times: A list of datetime objects representing the discrete
                      timestamps at which contexts are generated. Must not be empty.
      max_num_actions_per_ca: The maximum number of individual actions to include
                              in a single output `ContextualizedActions` object.
      max_num_cas_per_principal_snapshot: The maximum number of
                                          `ContextualizedActions` objects to emit
                                          for a principal at a single snapshot time.
                                          If 0 or negative, all are kept.

  Returns:
      A list of `ContextualizedActions` objects.
  """
  if not snapshot_times:
      raise ValueError("snapshot_times must not be empty.")

  # Sort snapshots to enable efficient binary search for time lookups.
  sorted_snapshots = sorted(snapshot_times)
  snapshot_times_set = set(sorted_snapshots)

  grouped_data = collections.defaultdict(
      lambda: {'context_features': [], 'actions': []}
  )

  # Group contexts by principal and their snapshot time.
  for principal, context in contexts:
    snapshot_time = time_utils.convert_proto_time_to_time(context.valid_from)
    if snapshot_time not in snapshot_times_set:
      raise ValueError(
          f"Context valid_from time {snapshot_time} not in snapshot_times"
      )
    key = (principal, snapshot_time)
    grouped_data[key]['context_features'].extend(context.features_per_source)
  
  # Group actions with the appropriate context snapshot.
  for source, principal_action_list in actions.items():
    for principal, action in principal_action_list:
      # Find the latest snapshot time that is less than or equal to the
      # action's occurrence time.
      idx = bisect.bisect_right(
          sorted_snapshots, time_utils.convert_proto_time_to_time(action.occurred_at))
      
      if idx == 0:
        # This action occurred before the earliest snapshot and cannot be
        # contextualized, so it's dropped.
        continue
      
      snapshot_time = sorted_snapshots[idx - 1]
      key = (principal, snapshot_time)
      grouped_data[key]['actions'].append((source, action))
  
  final_results = []
  cas_cap = (
      max_num_cas_per_principal_snapshot
      if max_num_cas_per_principal_snapshot > 0
      else float('inf')
  )

  for key, data in grouped_data.items():
    principal, snapshot_time = key
    context_features = data['context_features']
    associated_actions = data['actions']

    # If there are no context features for this key, drop associated actions.
    if not context_features:
      continue

    # Create the definitive context, checking for duplicate source types.
    seen_sources = set()
    for feature in context_features:
      if feature.source_type in seen_sources:
        raise ValueError(
            f"Duplicate source type '{feature.source_type}' for "
            f"principal '{principal}' at {snapshot_time}."
        )
      seen_sources.add(feature.source_type)
    
    final_context = contextualized_actions_pb2.FeaturizedContext(
        valid_from=time_utils.convert_to_proto_time(snapshot_time),
        features_per_source=context_features
    )

    # Handle context-only case: no actions to process.
    if not associated_actions:
      final_results.append(
          contextualized_actions_pb2.ContextualizedActions(
            principal=principal, context=final_context)
      )
      continue
    
    # Handle context-with-actions case, applying chunking logic.
    cas_emitted_for_key = 0
    current_actions_map = collections.defaultdict(list)
    action_count_in_current_ca = 0

    for source, action in associated_actions:
      current_actions_map[source].append(action)
      action_count_in_current_ca += 1

      if action_count_in_current_ca == max_num_actions_per_ca:
        # Sort actions by source type before adding to the object.
        sorted_actions = sorted([
            contextualized_actions_pb2.FeaturizedActionsBySource(
              source_type=s_type, actions=s_actions)
            for s_type, s_actions in current_actions_map.items()
        ], key=lambda x: x.source_type)
        
        final_results.append(contextualized_actions_pb2.ContextualizedActions(
            principal=principal,
            context=final_context,
            actions=sorted_actions
        ))
        cas_emitted_for_key += 1

        if cas_emitted_for_key >= cas_cap:
          break  # Stop processing actions for this key if cap is reached.

        # Reset for the next chunk.
        current_actions_map.clear()
        action_count_in_current_ca = 0
    
    # After the loop, emit the final (potentially partial) chunk of actions.
    if cas_emitted_for_key < cas_cap and action_count_in_current_ca > 0:
      sorted_actions = sorted([
          contextualized_actions_pb2.FeaturizedActionsBySource(
            source_type=s_type, actions=s_actions)
          for s_type, s_actions in current_actions_map.items()
      ], key=lambda x: x.source_type)
      final_results.append(contextualized_actions_pb2.ContextualizedActions(
          principal=principal,
          context=final_context,
          actions=sorted_actions
      ))

  return final_results
      
