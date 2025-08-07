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
"""Flattened multi-similarity loss."""

import tensorflow as tf
from tensorflow import keras


def _assert_float_tensor(t: tf.Tensor):
  if t.dtype not in [tf.float32, tf.float64]:
    raise ValueError('Expected float-type tensor. Got: {}'.format(t))


class MultiSimilarityLoss(keras.layers.Layer):
  r"""Flattened multi-similarity loss.

  Unlike the original paper (https://arxiv.org/abs/1904.06627), we don't really
  have a query-to-multiple-candidate-items setup. Instead, we just have positive
  and negative pairs. This implementation is therefore a flattened version,
  without per-query losses. For a dataset of {y_positives} and {y_negatives}
  scores, the loss is defined as:

  loss = loss_positives + loss_negatives

  where

  loss_positives = 1/a * log(1 + sum_{y \in y_positives} exp(-a * (y - loc))),
  loss_negatives = 1/b * log(1 + sum_{y \in y_negatives} exp(b * (y - loc))).

  a, b, and loc are parameters. a and b effectively control the strength of
  hard positive and negative mining respectively.
  """

  def __init__(self, a: float = 1.0, b: float = 1.0, loc: float = 0.0):
    """Initializes the loss."""
    super().__init__()
    assert a > 0
    assert b > 0
    self.a = a
    self.b = b
    self.loc = loc

  def get_config(self) -> dict[str, float]:
    return {
        'a': self.a,
        'b': self.b,
        'loc': self.loc,
    }

  def call(
      self,
      inputs: (
          tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor]
      ),
  ) -> tf.Tensor:
    """Computes the loss.

    Args:
      inputs: Either a pair or a 3-element tuple of rank 1 tensors, each
        respectively representing the binary labels, instance scores and
        instance weights. The labels are a float tensor such that values greater
        than 0 represent positive labels. All tensors must have the same size.

    Returns:
      loss: a scalar non-negative tensor holding the loss.
    """
    labels = inputs[0]
    scores = inputs[1]
    weights = inputs[2] if len(inputs) > 2 else None
    _assert_float_tensor(scores)
    _assert_float_tensor(labels)

    a = self.a
    b = self.b
    loc = self.loc

    y_pos_logits = -a * (tf.boolean_mask(scores, labels > 0.0) - loc)
    y_neg_logits = b * (tf.boolean_mask(scores, labels <= 0.0) - loc)
    if weights is not None:
      # This is the only tricky part: fold the instance weights into the
      # log-sum-exp.
      log_pos_weights = tf.math.log(tf.boolean_mask(weights, labels > 0.0))
      y_pos_logits += log_pos_weights
      log_neg_weights = tf.math.log(tf.boolean_mask(weights, labels <= 0.0))
      y_neg_logits += log_neg_weights

    y_pos_logits = tf.pad(y_pos_logits, [[1, 0]])  # Add a leading 0.
    loss_positives = 1 / a * tf.math.reduce_logsumexp(y_pos_logits)
    y_neg_logits = tf.pad(y_neg_logits, [[1, 0]])
    loss_negatives = 1 / b * tf.math.reduce_logsumexp(y_neg_logits)

    return loss_positives + loss_negatives
