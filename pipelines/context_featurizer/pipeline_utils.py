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

"""Utility functions for building context features."""

import bisect
import dataclasses
import datetime
import math
from typing import Dict, Iterable, List, Tuple

from common import time_utils
from context.graph.fold import Edge
from context.graph.fold import two_hops_random_walk_neighbors
from protos import context_pb2
from protos import context_source_config_pb2
from protos import contextualized_actions_pb2

dataclass = dataclasses.dataclass


def get_feature_name(
    base_name: str,
    mode: context_source_config_pb2.BipartiteGraph.TraversalMode,
) -> str:
  """Constructs a feature name based on the base name and traversal mode.

  Args:
      base_name: The base name of the feature.
      mode: The traversal mode.

  Returns:
      The constructed feature name.

  Raises:
      ValueError: If an unsupported traversal mode is provided.
  """

  if mode == context_source_config_pb2.BipartiteGraph.TraversalMode.TM_BACKWARD:
    return f"{base_name}_b"
  elif (
      mode == context_source_config_pb2.BipartiteGraph.TraversalMode.TM_FORWARD
  ):
    return f"{base_name}_f"
  elif (
      mode
      == context_source_config_pb2.BipartiteGraph.TraversalMode.TM_UNDIRECTED
  ):
    return f"{base_name}_u"
  else:
    raise ValueError("Unsupported traversal mode is chosen.")


def compute_discount(
    attribute_time: datetime.datetime,
    snapshot_time: datetime.datetime,
    half_life: datetime.timedelta,
) -> float:
  """Computes the discount factor based on the time difference and half-life.

  Args:
      attribute_time: The time of the attribute.
      snapshot_time: The time of the snapshot.
      half_life: The half-life duration.

  Returns:
      The computed discount factor.

  Raises:
      ValueError: If a negative half-life is provided.
  """

  if half_life < datetime.timedelta(0):
    raise ValueError("Negative half_life")

  if half_life == datetime.timedelta(0):
    return 1.0

  return math.pow(0.5, abs(attribute_time - snapshot_time) / half_life)


Node = Tuple[datetime.datetime, str]

PRINCIPAL_SUFFIX = "_p"
ATTRIBUTE_SUFFIX = "_a"


@dataclass(frozen=True)
class CommonContext:
  snapshot_time: datetime.datetime
  name: str
  value: str
  principal: str
  direction: context_pb2.PeerAttribute.Direction


ValueType = Tuple[datetime.datetime, str, float]


@dataclass(frozen=True)
class KeyedAttribute:
  key: CommonContext
  value: ValueType


def preprocess(
    feature_directive: Dict[str, context_source_config_pb2.PeerFeatureConfig],
    snapshot_intervals: List[Tuple[datetime.datetime, datetime.datetime]],
    source_type: str,
    context: context_pb2.Context,
) -> Iterable[KeyedAttribute]:
  """Preprocesses a context to generate keyed attributes for further processing.

  This function iterates through peer attributes of a given context, filters
  them based on the provided feature directive, and generates keyed attributes
  for each valid attribute. It considers snapshot intervals to determine
  relevant snapshot times for each attribute.

  Args:
      feature_directive: A dictionary mapping attribute names to
        PeerFeatureConfig.
      snapshot_intervals: A list of tuples, where each tuple represents a
        snapshot interval (start time, end time).
      source_type: The expected source type of the context.
      context: The context to be preprocessed.

  Yields:
      KeyedAttribute: An object containing a CommonContext key and a ValueType
        value for each valid peer attribute.

  Raises:
      ValueError: If the context source type does not match the expected type.
  """
  if context.type != source_type:
    raise ValueError(
        f"Context source types do not match. Config type: {source_type}"
        f" Context type: {context.type}"
    )

  context_time = time_utils.convert_proto_time_to_time(context.valid_from)
  featurization_begin = snapshot_intervals[0][0]
  featurization_end = snapshot_intervals[-1][1]

  if context_time < featurization_begin or context_time >= featurization_end:
    return

  for peer_attribute in context.peer_attributes:
    feature_config = feature_directive.get(peer_attribute.name)
    if feature_config is None:
      continue

    attribute_time = (
        time_utils.convert_proto_time_to_time(peer_attribute.time)
        if peer_attribute.HasField("time")
        else context_time
    )

    snapshot_times = get_snapshot_times(context_time, snapshot_intervals)
    if not snapshot_times:
      continue

    attribute_key_value = peer_attribute.value
    attribute_value = ""

    if (
        feature_config.aggregation_method
        == context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod.AGG_LATEST
    ):
      attribute_value = attribute_key_value
      attribute_key_value = ""

    for ts in snapshot_times:
      # Construct the CommonContext key
      key = CommonContext(
          snapshot_time=ts,
          name=peer_attribute.name,
          value=attribute_key_value,
          principal=context.principal,
          direction=peer_attribute.direction,
      )

      value_tuple: ValueType = (
          attribute_time,
          attribute_value,
          peer_attribute.weight,
      )

      yield KeyedAttribute(key=key, value=value_tuple)


def select_latest_attribute(
    item: Tuple[CommonContext, Iterable[ValueType]],
) -> KeyedAttribute:
  """Selects the latest attribute from an iterable of values.

  This function takes a tuple containing a CommonContext key and an iterable of
  ValueType values, and selects the value with the latest timestamp. It ensures
  that there is only one latest attribute.

  Args:
      item: A tuple containing a CommonContext key and an iterable of ValueType
        values.

  Returns:
      KeyedAttribute: An object containing the CommonContext key and the latest
        ValueType value.

  Raises:
      ValueError: If there is ambiguity in selecting the latest attribute.
  """
  key, values = item

  latest_attribute: ValueType = None
  counter = 0
  latest_ts = time_utils.time_ftz(datetime.MINYEAR, 1, 1, 0, 0, 0)

  for ts, value, weight in values:
    if ts == latest_ts:
      counter += 1
      continue
    if ts > latest_ts:
      latest_ts = ts
      latest_attribute = (ts, value, weight)
      counter = 1

  if counter != 1:
    raise ValueError(
        "Ambiguous AGG_LATEST attribute value aggregation"
        f" for attribute {key.name}"
    )

  common = CommonContext(
      snapshot_time=key.snapshot_time,
      name=key.name,
      value=latest_attribute[1],
      principal=key.principal,
      direction=key.direction,
  )

  return KeyedAttribute(key=common, value=(latest_ts, "", latest_attribute[2]))


def reduce(
    feature_directive: Dict[str, context_source_config_pb2.PeerFeatureConfig],
    source_type: str,
    item: Tuple[CommonContext, Iterable[ValueType]],
) -> Iterable[Tuple[CommonContext, ValueType]]:
  """Reduces a list of attribute values into a single context based on the feature directive.

  This function takes a CommonContext key and an iterable of ValueType values,
  and reduces them into a single context by applying the specified aggregation
  and weighting methods from the feature directive. It calculates the final
  weight for the context and yields the resulting context.

  Args:
      feature_directive: A dictionary mapping attribute names to
        PeerFeatureConfig.
      source_type: The expected source type of the context.
      item: A tuple containing a CommonContext key and an iterable of ValueType
        values.

  Yields:
      Tuple[CommonContext, ValueType]: A tuple containing the CommonContext key
        and the reduced ValueType value.

  Raises:
      ValueError: If no feature config is found for the attribute name or if an
        unsupported edge weighting method is chosen.
  """
  key, values = item
  snapshot_time = key.snapshot_time

  feature_config = feature_directive.get(key.name)
  if not feature_config:
    raise ValueError(f"No feature config found for {key.name}")

  graph_config = feature_config.bipartite_graph

  half_life = (
      time_utils.convert_proto_duration_to_timedelta(graph_config.half_life)
      if graph_config.HasField("half_life")
      else datetime.timedelta(0)
  )

  weight = 0.0
  attribute_time = time_utils.time_ftz(datetime.MINYEAR, 1, 1, 0, 0, 0)

  edge_weighting_method = graph_config.edge_weighting_method
  bipartite_graph = context_source_config_pb2.BipartiteGraph

  if edge_weighting_method in [
      bipartite_graph.EWM_LATEST,
      bipartite_graph.EWM_DISCOUNTED_LATEST,
  ]:
    for ts, _, ws in values:
      if ts > attribute_time:
        attribute_time = ts
        weight = ws
  elif edge_weighting_method in [
      bipartite_graph.EWM_SUM_DISCOUNTED,
      bipartite_graph.EWM_LOG_SUM_DISCOUNTED,
  ]:
    for ts, _, ws in values:
      weight += ws * compute_discount(ts, snapshot_time, half_life)
      if ts > attribute_time:
        attribute_time = ts
  else:
    raise ValueError("Unsupported edge weighting method is chosen.")

  if edge_weighting_method == bipartite_graph.EWM_DISCOUNTED_LATEST:
    weight *= compute_discount(attribute_time, snapshot_time, half_life)
  if edge_weighting_method == bipartite_graph.EWM_LOG_SUM_DISCOUNTED:
    weight = math.log(weight + 1.0)  # + 1.0 for positivity.

  result = context_pb2.Context()
  result.type = source_type
  result.principal = key.principal
  result.valid_from.CopyFrom(time_utils.convert_to_proto_time(snapshot_time))

  peer_attribute = result.peer_attributes.add()
  peer_attribute.name = key.name
  peer_attribute.value = key.value
  peer_attribute.direction = key.direction
  peer_attribute.weight = weight
  yield result


def is_direction_compatible(
    direction: context_pb2.PeerAttribute.Direction,
    traversal_mode: context_source_config_pb2.BipartiteGraph.TraversalMode,
) -> bool:
  if direction == context_pb2.PeerAttribute.Direction.D_UNSET:
    return (
        traversal_mode
        == context_source_config_pb2.BipartiteGraph.TraversalMode.TM_UNDIRECTED
    )
  return True


def make_edge(
    snapshot_time: datetime.datetime,
    principal: str,
    attribute: str,
    weight: float,
    is_forward: bool,
) -> Edge:
  """Creates an edge for the bipartite graph.

  Args:
      snapshot_time: The time of the snapshot.
      principal: The principal node.
      attribute: The attribute node.
      weight: The weight of the edge.
      is_forward: Whether the edge is forward or backward.

  Returns:
      The created edge.
  """
  if is_forward:
    return Edge(
        i=(snapshot_time, principal),
        j=(snapshot_time, attribute),
        weight=weight,
        is_middle=False,
    )
  else:
    return Edge(
        i=(snapshot_time, attribute),
        j=(snapshot_time, principal),
        weight=weight,
        is_middle=True,
    )


def flip_edge(edge: Edge) -> Edge:
  return Edge(
      i=edge.j, j=edge.i, weight=edge.weight, is_middle=not edge.is_middle
  )


def build_bipartite_graph(
    traversal_mode: context_source_config_pb2.BipartiteGraph.TraversalMode,
    context: context_pb2.Context,
) -> List[Edge]:
  """Builds edges for the bipartite graph based on the traversal mode and context.

  Args:
      traversal_mode: The traversal mode to use.
      context: The context to build the graph from.

  Returns:
      A list of edges for the bipartite graph.

  Raises:
      ValueError: If the context has an incompatible traversal mode.
      TypeError: If an unsupported traversal mode is chosen.
  """
  peer_attribute = context.peer_attributes[0]

  direction = peer_attribute.direction

  if not is_direction_compatible(direction, traversal_mode):
    raise ValueError(
        f"Context {context} has incompatible traversal mode with"
        f" {traversal_mode}"
    )

  weight = peer_attribute.weight
  if weight <= 0.0:
    return

  snapshot_time = time_utils.convert_proto_time_to_time(context.valid_from)

  principal = f"{context.principal}{PRINCIPAL_SUFFIX}"
  attribute = f"{peer_attribute.value}{ATTRIBUTE_SUFFIX}"

  bipartite_graph = context_source_config_pb2.BipartiteGraph

  if traversal_mode == bipartite_graph.TM_FORWARD:
    edge = make_edge(
        snapshot_time,
        principal,
        attribute,
        weight,
        direction != context_pb2.PeerAttribute.Direction.D_BACKWARD,
    )
    return [edge]
  elif traversal_mode == bipartite_graph.TM_UNDIRECTED:
    edge = make_edge(
        snapshot_time,
        principal,
        attribute,
        weight,
        direction != context_pb2.PeerAttribute.Direction.D_BACKWARD,
    )
    return [edge, flip_edge(edge)]
  elif traversal_mode == bipartite_graph.TM_BACKWARD:
    edge = make_edge(
        snapshot_time,
        principal,
        attribute,
        weight,
        direction != context_pb2.PeerAttribute.Direction.D_FORWARD,
    )
    return [edge]
  else:
    raise TypeError("Unsupported traversal mode chosen.")


def featurize_context(
    source_type: str,
    name: str,
    node: Tuple[Node, List[Tuple[float, Node]]],
) -> Tuple[str, contextualized_actions_pb2.FeaturizedContext]:
  """Featurizes a context node by converting its neighbors into a weighted bag of words.

  Args:
      source_type: The type of the context source.
      name: The name of the feature.
      node: A tuple containing the node (snapshot time and principal) and a list
        of weighted neighbor nodes.

  Returns:
      A tuple containing the principal name and the featurized context.
  """
  (snapshot_time, principal_p), neighbors = node

  principal = principal_p.removesuffix(PRINCIPAL_SUFFIX)

  result = contextualized_actions_pb2.FeaturizedContext()
  result.valid_from.CopyFrom(time_utils.convert_to_proto_time(snapshot_time))

  csf = result.features_per_source.add()
  csf.source_type = source_type

  feature = csf.features.add()
  feature.name = name

  weighted_words = feature.bag_of_weighted_words
  for weight, peer in neighbors:
    peer_name = peer[1].removesuffix(PRINCIPAL_SUFFIX)
    token = weighted_words.tokens.add()
    token.token = peer_name.encode("utf-8")
    token.weight = float(weight)

  return principal, result


def get_snapshot_times(
    context_time: datetime.datetime,
    snapshot_intervals: List[Tuple[datetime.datetime, datetime.datetime]],
) -> List[datetime.datetime]:
  """Finds the snapshot times that a given context time falls within.

  This function uses binary search to efficiently find the snapshot intervals
  that contain the given context time and returns the end times of those
  intervals.

  Args:
      context_time: The datetime of the context.
      snapshot_intervals: A list of tuples, where each tuple represents a
        snapshot interval (start time, end time).

  Returns:
      A list of datetime objects representing the end times of the snapshot
      intervals that the context time falls within.
  """
  lower_idx = bisect.bisect_right(
      snapshot_intervals, context_time, key=lambda interval: interval[1]
  )

  upper_idx = bisect.bisect_right(
      snapshot_intervals, context_time, key=lambda interval: interval[0]
  )

  return [interval[1] for interval in snapshot_intervals[lower_idx:upper_idx]]


def reduce_bipartite_peer_attributes(
    contexts: Iterable[context_pb2.Context],
    context_source_config: context_source_config_pb2.ContextSourceConfig,
    snapshot_times: List[datetime.datetime],
) -> Iterable[context_pb2.Context]:
  """Reduces bipartite peer attributes based on the given context source config.

  This function processes a list of contexts, reduces their peer attributes
  based on the provided configuration, and returns a list of reduced contexts.
  It handles different aggregation methods, time-based discounts, and
  snapshot intervals.

  Args:
      contexts: An iterable of context_pb2.Context objects.
      context_source_config: A context_source_config_pb2.ContextSourceConfig
        object.
      snapshot_times: A list of datetime objects representing snapshot times.

  Returns:
      An iterable of reduced context_pb2.Context objects.

  Raises:
      ValueError: If context_lookback is negative, snapshot_times is empty or
      not sorted, or if there are duplicate snapshot times.
  """

  lookback = time_utils.convert_proto_duration_to_timedelta(
      context_source_config.context_lookback
  )

  if lookback <= datetime.timedelta(0):
    raise ValueError("context_lookback must be a positive duration.")

  if not snapshot_times:
    raise ValueError("Empty snapshot_times to featurize contexts.")

  if not all(
      snapshot_times[i] <= snapshot_times[i + 1]
      for i in range(len(snapshot_times) - 1)
  ):
    raise ValueError("Input snapshot_times must be sorted.")

  if len(snapshot_times) != len(set(snapshot_times)):
    raise ValueError("Duplicate snapshot_times exist.")

  snapshot_intervals = [(ts - lookback, ts) for ts in snapshot_times]

  feature_directive: Dict[str, context_source_config_pb2.PeerFeatureConfig] = {}

  for config in context_source_config.peer_feature_configs:
    if config.HasField("bipartite_graph"):
      if (
          config.aggregation_method
          == context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod.AGG_UNSPECIFIED
      ):
        raise ValueError("Aggregation method must be specified.")
      feature_directive[config.name] = config

  source_type = context_source_config.type

  processed: List[KeyedAttribute] = []
  for context in contexts:
    processed.extend(
        preprocess(feature_directive, snapshot_intervals, source_type, context)
    )

  accumulated: List[KeyedAttribute] = []
  latest: List[KeyedAttribute] = []
  for item in processed:
    if item.key.value:
      accumulated.append(item)
    else:
      latest.append(item)

  # Group latest by key
  latest_grouped: Dict[CommonContext, List[ValueType]] = {}
  for item in latest:
    if item.key not in latest_grouped:
      latest_grouped[item.key] = []
    latest_grouped[item.key].append(item.value)

  latest_aggregated: List[KeyedAttribute] = []
  for key, values in latest_grouped.items():
    latest_aggregated.append(select_latest_attribute((key, values)))

  result: List[KeyedAttribute] = accumulated + latest_aggregated

  # Group result by key
  result_grouped: Dict[CommonContext, List[ValueType]] = {}
  for item in result:
    if item.key not in result_grouped:
      result_grouped[item.key] = []
    result_grouped[item.key].append(item.value)

  final_contexts: List[context_pb2.Context] = []
  for key, values in result_grouped.items():
    final_contexts.extend(reduce(feature_directive, source_type, (key, values)))

  return final_contexts


def featurize_bipartite_peer_attributes(
    contexts: Iterable[context_pb2.Context],
    context_source_config: context_source_config_pb2.ContextSourceConfig,
) -> List[Tuple[str, contextualized_actions_pb2.FeaturizedContext]]:
  """Featurizes bipartite peer attributes based on the given context source config."""

  source_type = context_source_config.type

  # Holding FeaturizedContext keyed by principal per PeerAttribute.name.
  featurized_context_vec: List[
      Dict[str, contextualized_actions_pb2.FeaturizedContext]
  ] = []

  for config in context_source_config.peer_feature_configs:
    if not config.HasField("bipartite_graph"):
      continue

    # Checks the number of peer_attributes and filters by config.name.
    filtered_contexts = []
    for context in contexts:
      if len(context.peer_attributes) != 1:
        raise ValueError(
            "Context must have only one peer_attribute at graph building stage."
        )
      if context.peer_attributes[0].name == config.name:
        filtered_contexts.append(context)

    if not filtered_contexts:
      continue

    for tm in config.bipartite_graph.traversal_modes:
      name = get_feature_name(config.name, tm)

      graph: List[Edge] = []

      for context in filtered_contexts:
        edges_to_add = build_bipartite_graph(tm, context) or []
        graph.extend(edges_to_add)

      # Folds the bipartite graph to connect principals to peers.
      folded_graph = two_hops_random_walk_neighbors(graph, config.max_peers)

      # Iterate through the dictionary items (Node, List of neighbors)
      for node, neighbors in folded_graph.items():
        principal, context = featurize_context(
            source_type, name, (node, neighbors)
        )

        featurized_context_vec.append((principal, context))

  return featurized_context_vec

