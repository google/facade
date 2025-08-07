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
"""Monotonically increasing transformations for model score rescaling."""

from collections.abc import Mapping
from typing import Any

import tensorflow as tf
from tensorflow import keras

from google.protobuf import json_format
from protos import config_pb2


class LinearRescaler(keras.layers.Layer):
  """Linear X -> aX + b transformation where a, b are scalars and a > 0."""

  def __init__(self, config: config_pb2.LossFunction.LinearRescalerConfig):
    """Builds the layer."""
    super().__init__()
    if config.scale <= 0:
      raise ValueError(f'Scale must be positive. Got: {config.scale}.')
    self.config = config
    self.a = tf.Variable(
        initial_value=config.scale, trainable=config.trainable_scale
    )
    self.b = tf.Variable(
        initial_value=config.offset, trainable=config.trainable_offset
    )

  def get_config(self) -> Mapping[str, Any]:
    return {'config': json_format.MessageToJson(self.config)}

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> 'LinearRescaler':
    cfg = config_pb2.LossFunction.LinearRescalerConfig()
    json_format.Parse(config['config'], cfg)
    return cls(cfg)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Applies the layer."""
    return tf.math.abs(self.a) * inputs + self.b
