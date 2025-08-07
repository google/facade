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
"""Converts variable-size feature segments into fixed-size embeddings."""

from collections.abc import Mapping
import math
from typing import Any, Optional, Union

import tensorflow as tf
from tensorflow import keras

from protos import config_pb2


class SegmentEmbedder(keras.layers.Layer):
  """Transforms a batch of raw feature segments into a numerical tensor.

  This operation is specific to Facade's hierarchical feature representation.

  For actions, the hierarchy is:
  A batch is made of actions
    -> an action is made of features
      -> a feature is made of varying-length segments of tokens, with
         optional intensities attached to each token. For instance, a
         drive_visited particle may contain 4 segment-features: usernames,
         costcenters, managers and locations.

  For contexts, the hierarchy is the same, but the actions dimension is not
  present, as contexts represent features at a single point in time.

  This module handles one segment-feature at a time (can be either action or
  context). It constructs numerical representations from low-level token
  embeddings and indexing information as follows. Token embeddings are averaged
  (optionally weight-averaged if `intensities` is set) within each feature. The
  averaging operation always includes a global offset token initialized at 0 and
  having a fixed weight of 1.

  When `intensities` is not set, the module assumes uniform unitary weights.
  For contexts, the result is a tf.Tensor with the shape
  [batch_dim, embedding_dim]. For actions, the result is a tf.RaggedTensor with
  shape `[batch_dim, <ragged_dim>, embedding_dim]`.
  """

  def __init__(
      self,
      weight_scaling: config_pb2.SegmentReduction.WeightScaling,
      weight_normalization: config_pb2.SegmentReduction.WeightNormalization,
      dropout_rate: float,
  ):
    """Constructs a SegmentEmbedder layer.

    Args:
      weight_scaling: Transformation to apply to the raw weights.
      weight_normalization: Normalization of the aggregated weighted embeddings.
      dropout_rate: At training time, randomly drops out tokens.
    """
    super().__init__()
    self._weight_scaling = weight_scaling
    self._weight_normalization = weight_normalization
    self._dropout_rate = dropout_rate

    self._offset_token_weight = 1.0
    if weight_scaling == config_pb2.SegmentReduction.WeightScaling.WS_LOG:
      self._offset_token_weight = math.log1p(self._offset_token_weight)

  def get_config(self) -> dict[str, Any]:
    return {
        'weight_scaling': self._weight_scaling,
        'weight_normalization': self._weight_normalization,
        'dropout_rate': self._dropout_rate,
    }

  def build(self, input_shapes: Mapping[str, tf.TensorShape]) -> None:
    """Constructs offset token embedding the first time call() is invoked."""
    embedding_dims = tf.cast(input_shapes['embeddings'][-1], tf.int32)
    self._offset_token = self.add_weight(
        name='offset_token',
        shape=[1, embedding_dims],
        initializer='zeros',
        dtype=tf.float32,
    )

  def call(
      self,
      inputs: Mapping[str, Union[tf.Tensor, tf.RaggedTensor]],
      training: Optional[bool] = None,
  ) -> Union[tf.Tensor, tf.RaggedTensor]:
    """Applies the layer to a batch of raw segment-features.

    The segment-feature is defined by the set of applicable RaggedTensors.

    Args:
      inputs: A dictionary of named RaggedTensors representing the segment.
        Expected key:values are:
        * 'embeddings': RaggedTensor holding the embeddings of the segment
          tokens.
        * [optional] intensities: RaggedTensor constructed from reading a batch
          of tf.SequenceExample-serialized Facade ContextualizedAction
          instances. Set to None for default implicit 1.0 weights. If provided,
          weights must be non-negative.
      training: Configures the graph for training (turning dropout on) as
        opposed to test/eval/inference.

    Returns:
      Per-example embeddings in a tf.RaggedTensor, for actions. Per-batch
      embeddings in a tf.RaggedTensor, for contexts.
    """  # fmt: skip
    embeddings = inputs['embeddings']
    if 'intensities' in inputs:
      intensities = inputs['intensities']
      tf.debugging.assert_non_negative(intensities.flat_values)
      intensities = tf.expand_dims(intensities, axis=-1)
    else:
      # 1.0 is the default intensity.
      ones = tf.ones(
          [tf.shape(embeddings.flat_values, out_type=tf.int32)[0], 1],
          dtype=tf.float32,
      )
      intensities = embeddings.with_flat_values(ones)

    if not intensities.dtype.is_floating:
      raise ValueError(
          f'intensities must be of type float. Got input: {intensities}.'
      )

    ws = config_pb2.SegmentReduction.WeightScaling  # formatter-friendly name.
    if self._weight_scaling == ws.WS_IDENTITY:
      intensities = tf.identity(intensities)
    elif self._weight_scaling == ws.WS_LOG:
      intensities = tf.math.log1p(intensities)
    elif self._weight_scaling == ws.WS_UNIFORM:
      intensities = tf.ones_like(intensities)
    else:
      raise ValueError(f'Unknown weight scaling: {self._weight_scaling}')

    # Dropout is achieved by zero-ing intensities. This is safe because we
    # always add the offset token to every segment, so the total intensity is
    # at least 1.0.
    if training and self._dropout_rate > 0:
      mask = tf.random.uniform(shape=tf.shape(intensities.flat_values))
      mask = mask > tf.cast(self._dropout_rate, tf.float32)
      mask = tf.cast(mask, intensities.dtype)
      intensities *= intensities.with_flat_values(mask)
    embeddings *= intensities

    # Compute summation domains.
    segment_embeddings = tf.math.reduce_sum(embeddings, axis=-2)

    # Add offset token to every embedding.
    segment_embeddings += self._offset_token_weight * self._offset_token
    # Normalize by total intensity.
    wn = config_pb2.SegmentReduction.WeightNormalization
    if self._weight_normalization == wn.WN_L1:
      segment_intensities = tf.math.reduce_sum(intensities, axis=-2)
      segment_intensities += self._offset_token_weight
    elif self._weight_normalization == wn.WN_L2:
      segment_intensities = tf.math.reduce_sum(
          tf.math.square(intensities), axis=-2
      )
      segment_intensities += self._offset_token_weight**2
      segment_intensities = tf.math.sqrt(segment_intensities)
    else:
      raise ValueError(
          f'Unknown weight normalization: {self._weight_normalization}'
      )
    segment_embeddings /= segment_intensities
    return segment_embeddings
