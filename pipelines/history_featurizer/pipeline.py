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


import collections
import datetime
import math
from typing import Dict, List, Tuple, Union, Optional

from protos import timestamp_pb2, contextualized_actions_pb2, action_pb2
from pipelines.history_featurizer.online_heaviest_items import OnlineHeaviestItems

# Proto aliases
Action = action_pb2.Action
Feature = contextualized_actions_pb2.Feature
WeightedToken = contextualized_actions_pb2.WeightedToken

K_PRINCIPAL_FEATURE = "prin"


# Helper function to convert proto timestamp to datetime
def proto_to_datetime(ts: timestamp_pb2.Timestamp) -> datetime.datetime:
  return datetime.datetime.fromtimestamp(ts.seconds, tz=datetime.timezone.utc)


# Helper function to create WeightedToken
def make_weighted_token(token: str, weight: float) -> WeightedToken:
  return WeightedToken(token=token, weight=weight)


def get_history_tokens(action: Action) -> List[Tuple[str, WeightedToken]]:
  """Extracts history tokens from an action."""
  if not action.principal:
    return []
  return [(
      K_PRINCIPAL_FEATURE,
      make_weighted_token(action.principal.encode("utf-8"), 1.0),
  )]


class HistoryFeaturizer:
  """
  Accumulates and deduplicates tokens over a rolling window to build features.
  """

  def __init__(
      self, action_deduplication_period: datetime.timedelta, max_tokens_per_segment: int
  ):
    self.action_deduplication_period = action_deduplication_period
    self.max_tokens_per_segment = max_tokens_per_segment
    self.segments: Dict[str, OnlineHeaviestItems] = {}
    self.last_seen: Dict[str, Dict[str, datetime.datetime]] = collections.defaultdict(dict)

  def accumulate(
      self, event_time: datetime.datetime, tokens: List[Tuple[str, WeightedToken]]
  ) -> None:
    """Temporally deduplicates tokens and tallies their intensities."""
    for key, token in tokens:
      if key not in self.segments:
        self.segments[key] = OnlineHeaviestItems(self.max_tokens_per_segment)

      segment = self.segments[key]
      last_seen_segment = self.last_seen[key]

      # Temporal deduplication: skips recently seen tokens.
      lookup_key = token.token.decode("utf-8", "ignore") + ("+" if token.weight > 0 else "-")
      if lookup_key in last_seen_segment:
        if (
            last_seen_segment[lookup_key] + self.action_deduplication_period
            >= event_time
        ):
          continue
      last_seen_segment[lookup_key] = event_time
      segment.upsert(token.token.decode("utf-8", "ignore"), token.weight)

  def make_tokens(self) -> List[Tuple[str, WeightedToken]]:
    """Creates the list of tokens from the current state."""
    result = []
    for key, segment in self.segments.items():
      for iw in segment.heaviest():
        result.append((key, make_weighted_token(iw.item.encode("utf-8"), iw.weight)))
    return result


def drop_too_frequent_actions(
    actions: List[Action],
    featurize_actions_at_or_after: datetime.datetime,
    featurize_actions_before: datetime.datetime,
    history_duration: datetime.timedelta,
    max_history_keys_per_day: int,
) -> List[Action]:
  """Drops history keys that are too frequent."""
  if max_history_keys_per_day < 0:
    return actions

  count_before = featurize_actions_before
  count_at_or_after = featurize_actions_at_or_after - history_duration

  max_history_keys = math.ceil(
      max_history_keys_per_day
      * (count_before - count_at_or_after).total_seconds()
      / (24 * 3600)
  )

  history_key_counts = collections.defaultdict(int)
  for a in actions:
    event_time = proto_to_datetime(a.occurred_at)
    if count_at_or_after <= event_time < count_before:
      history_key_counts[a.history_key] += 1

  not_too_frequent_keys = {
      key
      for key, count in history_key_counts.items()
      if count <= max_history_keys
  }

  featurize_actions = []
  not_featurize_actions = []
  for a in actions:
    event_time = proto_to_datetime(a.occurred_at)
    if (
        featurize_actions_at_or_after <= event_time < featurize_actions_before
    ):
      if a.history_key in not_too_frequent_keys:
        featurize_actions.append(a)
    else:
      not_featurize_actions.append(a)

  return featurize_actions + not_featurize_actions


def build_history_features(
    actions: List[Action],
    featurize_actions_at_or_after: datetime.datetime,
    featurize_actions_before: datetime.datetime,
    history_duration: datetime.timedelta,
    action_deduplication_period: datetime.timedelta,
    max_tokens_per_segment: int,
    max_history_keys_per_day: int = -1,
    min_tokens_per_segment: int = 0,
) -> Dict[str, List[Feature]]:
  """
  Builds history features for actions within the given time range.

  Args:
    actions: A list of Action protos.
    featurize_actions_at_or_after: Start time for featurizing actions.
    featurize_actions_before: End time for featurizing actions.
    history_duration: Duration of the history window.
    action_deduplication_period: Deduplication period for actions.
    max_tokens_per_segment: Maximum number of tokens per feature segment.
    max_history_keys_per_day: Maximum number of history keys per day.
    min_tokens_per_segment: Minimum number of tokens per feature segment.

  Returns:
    A dictionary mapping action IDs to a list of Feature protos.
  """

  actions = drop_too_frequent_actions(
      actions,
      featurize_actions_at_or_after,
      featurize_actions_before,
      history_duration,
      max_history_keys_per_day,
  )

  # Sort actions by time
  actions.sort(key=lambda a: proto_to_datetime(a.occurred_at))

  # Group actions by history key
  actions_by_target: Dict[str, List[Action]] = collections.defaultdict(list)
  for a in actions:
    actions_by_target[a.history_key].append(a)

  results: Dict[str, List[Feature]] = {}

  for target_key in actions_by_target:
    target_actions = actions_by_target[target_key]
    featurizer = HistoryFeaturizer(
        action_deduplication_period, max_tokens_per_segment
    )

    # Events to process for the current target: (time, type, data)
    # type: 0 for token accumulation, 1 for featurization request
    events = []

    for i, action in enumerate(target_actions):
      time = proto_to_datetime(action.occurred_at)
      if time >= featurize_actions_before:
        continue

      # Add event for featurizing this action
      if featurize_actions_at_or_after <= time < featurize_actions_before:
        events.append({"time": time, "type": 0, "payload": action})

      # Add events for history token accumulation
      # if not action.usage.ignore_for_history:
      raw_tokens = get_history_tokens(action)
      if raw_tokens:
        events.append({"time": time, "type": 1, "payload": raw_tokens})

        # Add events for token expiration
        if history_duration > datetime.timedelta(0):
          expiration_time = time + history_duration
          if expiration_time < featurize_actions_before:
            expiration_tokens = []
            for key, token in raw_tokens:
              exp_token = WeightedToken()
              exp_token.CopyFrom(token)
              exp_token.weight = -token.weight
              expiration_tokens.append((key, exp_token))
            events.append({
                "time": expiration_time,
                "type": 1,
                "payload": expiration_tokens,
            })


    # Sort events by time, then type (accumulate before featurize at same time)
    events.sort(key=lambda e: (e["time"], e["type"]))

    for event in events:
      event_time = event["time"]
      event_type = event["type"]
      payload = event["payload"]

      if event_type == 0:  # Featurize action
        action_to_featurize = payload
        current_tokens = featurizer.make_tokens() # Tokens from *before* this action

        feature = Feature(name=K_PRINCIPAL_FEATURE)
        bow = feature.bag_of_weighted_words
        for key, token in current_tokens:
          if key == K_PRINCIPAL_FEATURE:
            bow.tokens.append(token)

        if len(bow.tokens) >= min_tokens_per_segment:
          results[action_to_featurize.id.decode("utf-8", "ignore")] = [feature]

      elif event_type == 1:  # Accumulate tokens
        featurizer.accumulate(event_time, payload)

  return results


def get_feature_names() -> List[str]:
  """Returns the names of the features generated by this pipeline."""
  return [K_PRINCIPAL_FEATURE]