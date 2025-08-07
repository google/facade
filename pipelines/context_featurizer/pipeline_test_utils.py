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

"""Utility functions for testing the pipeline."""

import collections
import datetime
from typing import List
from common import time_utils
from google.protobuf import json_format
from protos import context_pb2
from protos import context_source_config_pb2

MessageToDict = json_format.MessageToDict


def make_context(
    principal: str,
    valid_from: datetime.datetime,
    peer_attributes: List[context_pb2.PeerAttribute] | None = None,
    context_type: str = "test",
) -> context_pb2.Context:
  """Creates a context proto.

  Args:
    principal: The principal of the context.
    valid_from: The valid from time of the context.
    peer_attributes: The peer attributes of the context.
    context_type: The type of the context.

  Returns:
    A context proto.
  """
  context = context_pb2.Context(
      principal=principal,
      valid_from=time_utils.convert_to_proto_time(valid_from),
      type=context_type,
  )
  if peer_attributes is not None:
    context.peer_attributes.extend(peer_attributes)
  return context


def make_peer_attribute(
    name: str,
    value: str,
    weight: float,
    time: datetime.datetime,
    direction: context_pb2.PeerAttribute.Direction | None = None,
) -> context_pb2.PeerAttribute:
  """Creates a peer attribute proto.

  Args:
    name: The name of the peer attribute.
    value: The value of the peer attribute.
    weight: The weight of the peer attribute.
    time: The time of the peer attribute.
    direction: The direction of the peer attribute.

  Returns:
    A peer attribute proto.
  """
  peer_attribute = context_pb2.PeerAttribute(
      name=name.encode("utf-8"),
      value=value.encode("utf-8"),
      weight=weight,
      time=time_utils.convert_to_proto_time(time),
  )
  if direction is not None:
    peer_attribute.direction = direction
  return peer_attribute


def make_context_source_config(
    lookback: datetime.timedelta,
    peer_feature_configs: List[context_source_config_pb2.PeerFeatureConfig],
    context_type: str = "test",
) -> context_source_config_pb2.ContextSourceConfig:
  """Creates a context source config proto.

  Args:
    lookback: The lookback duration of the context source.
    peer_feature_configs: The peer feature configs of the context source.
    context_type: The type of the context source.

  Returns:
    A context source config proto.
  """
  config = context_source_config_pb2.ContextSourceConfig(
      context_lookback=time_utils.convert_to_proto_duration(lookback),
      type=context_type,
  )
  config.peer_feature_configs.extend(peer_feature_configs)
  return config


def make_bipartite_graph_config(
    traversal_modes: List[
        context_source_config_pb2.BipartiteGraph.TraversalMode
    ],
    half_life: datetime.timedelta,
    method: context_source_config_pb2.BipartiteGraph.EdgeWeightingMethod,
) -> context_source_config_pb2.BipartiteGraph:
  """Creates a bipartite graph config proto.

  Args:
    traversal_modes: The traversal modes of the bipartite graph.
    half_life: The half life of the bipartite graph.
    method: The edge weighting method of the bipartite graph.

  Returns:
    A bipartite graph config proto.
  """
  graph = context_source_config_pb2.BipartiteGraph(
      half_life=time_utils.convert_to_proto_duration(half_life),
      edge_weighting_method=method,
  )
  graph.traversal_modes.extend(traversal_modes)
  return graph


def make_peer_feature_config(
    name: str,
    max_peers: int,
    graph: context_source_config_pb2.BipartiteGraph | None = None,
    agg_method: (
        context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod
        | None
    ) = None,
) -> context_source_config_pb2.PeerFeatureConfig:
  """Creates a peer feature config proto.

  Args:
    name: The name of the peer feature.
    max_peers: The maximum number of peers to consider.
    graph: The bipartite graph config for the peer feature.
    agg_method: The aggregation method for the peer feature.

  Returns:
    A peer feature config proto.
  """
  config = context_source_config_pb2.PeerFeatureConfig(
      name=name.encode("utf-8"),
      max_peers=max_peers,
  )
  if graph is not None:
    config.bipartite_graph.CopyFrom(graph)
  if agg_method is not None:
    config.aggregation_method = agg_method
  return config


def assert_proto2_equal_ignoring_fields(
    proto1, proto2, ignored_fields=None, ignore_order_in_fields=None
):
  """Compares two protobufs, ignoring specified fields and repeated field order.

  Args:
      proto1: The first protobuf message.
      proto2: The second protobuf message.
      ignored_fields: A list of field names to ignore during comparison.
      ignore_order_in_fields: A list of repeated field names where
        element order should be ignored.

  Returns:
      bool: True if the messages are considered equal, False otherwise.
  """
  # Use preserving_proto_field_name=True to keep original field names
  dict1 = MessageToDict(proto1, preserving_proto_field_name=True)
  dict2 = MessageToDict(proto2, preserving_proto_field_name=True)

  # Remove fields that should be ignored from both dictionaries
  if ignored_fields:
    for field in ignored_fields:
      dict1.pop(field, None)
      dict2.pop(field, None)

  # Convert lists where order doesn't matter into Counter objects for comparison
  if ignore_order_in_fields:
    for field in ignore_order_in_fields:
      if field in dict1 and isinstance(dict1.get(field), list):
        hashable_list1 = [tuple(sorted(d.items())) for d in dict1[field]]
        dict1[field] = collections.Counter(hashable_list1)

      if field in dict2 and isinstance(dict2.get(field), list):
        hashable_list2 = [tuple(sorted(d.items())) for d in dict2[field]]
        dict2[field] = collections.Counter(hashable_list2)

  # Finally, compare the processed dictionaries
  return dict1 == dict2

