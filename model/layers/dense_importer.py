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
"""Simple identity function that verifies dense vectors have the right dims."""

from collections.abc import Mapping
from typing import Union

import tensorflow as tf
from tensorflow import keras


class DenseImporter(keras.layers.Layer):
  """Ensures dense vectors have the right dimensionality.

  For actions, inputs are ragged and follow the hierarchy:
  A batch is made of actions
    -> an action is made of features
      -> features are wrapped in a size 1 dimension
        -> a feature is made of a fixed-dimension vector of floats
  This pipeline removes the extra size 1 dimension from actions.

  For contexts, inputs are fixed tensors with just a batch and dense dimension.

  The input vectors are checked against the dense_vector_size input as a
  validation.
  """

  def __init__(
      self,
      dense_vector_size: int,
  ):
    """Constructs a DenseImporter layer.

    Args:
      dense_vector_size: The number of dimensions of the imported dense vector.
    """
    super().__init__()
    self._dense_vector_size = dense_vector_size

  def get_config(self) -> dict[str, int]:
    return {
        'dense_vector_size': self._dense_vector_size,
    }

  def call(
      self,
      inputs: Mapping[str, Union[tf.Tensor, tf.RaggedTensor]],
  ) -> Union[tf.Tensor, tf.RaggedTensor]:
    """Applies the layer to a batch of raw segment-features.

    Args:
      inputs: A dictionary of named Tensors for contexts, or RaggedTensors for
        actions representing the segment. Expected key:values are: *
        'dense_vector': Ragged tensor holding the dense vectors of the segment.

    Returns:
      A Tensor for contexts, or a RaggedTensor for actions.
    """
    dense_vector = inputs['dense_vector']

    # Input pipeline adds an extra empty dimension for actions just above the
    # features. Check if this is an action input by looking at the number of
    # dimensions.
    if len(dense_vector.shape) == 4:
      dense_vector = tf.squeeze(dense_vector, axis=2)

    if dense_vector.shape[-1] != self._dense_vector_size:
      raise ValueError(
          'Input dense vector shape is not compatible with the dense vector '
          f'size set in the config file: {dense_vector.shape[-1]} vs. '
          f'{self._dense_vector_size}!'
      )

    return dense_vector
