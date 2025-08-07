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
"""Library for reading actions and contexts from TFRecord files."""

from typing import List
import tensorflow as tf
import datetime
from common import time_utils

from protos import action_pb2
from protos import context_pb2


def read_actions(action_type: str, tf_record_file: str, start_time: datetime.datetime, end_time: datetime.datetime) -> List[action_pb2.Action]:
  """Reads actions of the given type from the TFRecord file."""
  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tf_record_file))
  actions = []
  for raw_record in dataset:
    action = action_pb2.Action()
    action.ParseFromString(raw_record.numpy())
    occurred_at = time_utils.convert_proto_time_to_time(action.occurred_at)
    if (action.type == action_type) and occurred_at >= start_time and occurred_at < end_time:
      actions.append(action)
  return actions


def read_context(context_type: str, tf_record_file: str, start_time: datetime.datetime, end_time: datetime.datetime) -> List[context_pb2.Context]:
  """Reads contexts of the given type from the TFRecord file."""
  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tf_record_file))
  contexts = []
  for raw_record in dataset:
    context = context_pb2.Context()
    context.ParseFromString(raw_record.numpy())
    valid_from = time_utils.convert_proto_time_to_time(context.valid_from)
    if (context.type == context_type) and valid_from >= start_time and valid_from < end_time:
      contexts.append(context)
  return contexts
