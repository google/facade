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
"""Provides an OovDropout layer."""

import tensorflow as tf

AnyTensor = tf.Tensor | tf.SparseTensor | tf.RaggedTensor


class OovDropout(tf.keras.layers.Layer):
  """When training, converts looked-up string indices to oov-token indices.

  This class assumes its called on a tensor that was output from the keras
  StringLookup layer, running in output='int' mode with no mask token.
  """

  def __init__(self, dropout_prob: float, num_oov_tokens: int, **kwargs):
    """Initializes the class.

    Args:
      dropout_prob: Probability of converting a string to a random one.
      num_oov_tokens: Number of out of vocab tokens used.
      **kwargs: Passed to keras Layer.
    """
    super().__init__(trainable=False, **kwargs)
    if num_oov_tokens < 1:
      raise ValueError("There must be at least one OOV token.")
    if not 0 <= dropout_prob < 1:
      raise ValueError(f"dropout_prob must be in [0,1), found: {dropout_prob}")
    self._num_oov_tokens = num_oov_tokens
    self._dropout_prob = dropout_prob

  def call(self, x: AnyTensor, training: bool = False) -> AnyTensor:
    if x.dtype not in (tf.int32, tf.int64):
      raise ValueError(
          "OovDropout must be called on int tensors (after StringLookup is "
          "applied)."
      )

    if not training:
      return x

    # Flatten or extract values of tensors so this class works for ragged,
    # sparse, and regular tensors.
    if isinstance(x, tf.RaggedTensor):
      flat = x.flat_values
    elif isinstance(x, tf.SparseTensor):
      flat = x.values
    else:
      flat = x
    flat_shape = tf.shape(flat)

    # Assuming no mask token, the first num_oov_tokens are OOV, followed by the
    # normal vocabulary.
    # https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup
    oov_tokens = tf.random.uniform(
        flat_shape, minval=0, maxval=self._num_oov_tokens, dtype=x.dtype
    )
    should_drop = (
        tf.random.uniform(flat_shape, minval=0, maxval=1, dtype=tf.float32)
        < self._dropout_prob
    )
    output = tf.where(should_drop, oov_tokens, flat)

    if isinstance(x, tf.RaggedTensor):
      return x.with_flat_values(output)
    elif isinstance(x, tf.SparseTensor):
      return x.with_values(output)
    else:
      return output
