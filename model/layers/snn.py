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
"""Self-normalizing feedforward neural network with optional alpha-dropout."""

from collections.abc import Sequence
from typing import Any, Optional

import tensorflow as tf
from tensorflow import keras


class SNN(keras.layers.Layer):
  """Self-normalizing feedforward neural network with optional alpha-dropout.

  The sequence of operations is:
    linear -> selu [-> dropout] -> ...
    linear -> selu [-> dropout] -> linear
  where the number of linear mappings define the number of layers. Neither
  dropout nor selu is applied after the last linear mapping.
  """

  def __init__(self, layer_sizes: Sequence[int], dropout_rate: float):
    """Initializes an SNN module.

    Args:
      layer_sizes: The sizes of the consecutive linear transformations. The last
        entry of layer_sizes is output dimension.
      dropout_rate: At training time, applies Alpha Dropout to the hidden
        layers.
    """
    super().__init__()
    self._layer_sizes = layer_sizes
    self._dropout_rate = dropout_rate
    if not layer_sizes:
      raise ValueError('Need at least one layer.')
    self.layers = []
    for size in layer_sizes[:-1]:
      self.layers.append(keras.layers.Dense(size, activation='selu'))
      self.layers.append(keras.layers.AlphaDropout(dropout_rate))
    self.layers.append(keras.layers.Dense(layer_sizes[-1]))

  def get_config(self) -> dict[str, Any]:
    return {
        'layer_sizes': self._layer_sizes,
        'dropout_rate': self._dropout_rate,
    }

  def call(
      self, inputs: tf.Tensor, training: Optional[bool] = None
  ) -> tf.Tensor:
    """Applies the SNN to the input.

    Args:
      inputs: A 2D tensor where the first dimension is the batch dimension.
      training: Configures the graph for training (turning dropout on) as
        opposed to test/eval/inference.

    Returns:
      y: A 2D tensor with shape `[inputs.shape[0], layer_sizes[-1]]`.
    """
    y = inputs
    for f in self.layers:
      y = f(y, training=training)
    return y
