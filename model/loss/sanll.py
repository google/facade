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
"""SANLL loss: Score As Negative Log Likelihood (of negative instance)."""

import tensorflow as tf
from tensorflow import keras

_EPSILON = 1e-9


def _assert_float_tensor(t: tf.Tensor):
  if t.dtype not in [tf.float32, tf.float64]:
    raise ValueError('Expected float-type tensor. Got: {}'.format(t))


class SanllLoss(keras.losses.Loss):
  """SANLL loss: Score As Negative Log Likelihood of being in the negative class.

  This pointwise loss encodes log-likelihood minimization by interpreting
  the score of each instance as the negative log-likelihood of the instance
  belonging to the natural (negative) class. Therefore, the loss of a negative
  instance is simply its score, and the loss of a positive instance is
  `-log(1 - exp(-score))`.

  For this loss to make sense, the scores must be non-negative.

  Two additional parameters m and p respectively control the effective margin,
  or scaling of the loss, and its bias towards penalizing higher scoring
  negative instances. p=0 and m=1 recover the regular Sanll loss introduced
  above. Mathematically, we have:

  For a negative instance:
    loss(score) = m * ((1 + score / m)^(1 + p) - 1)
  For a positive instance:
    loss(score) = -m * log(1 - exp(-score / m)).
  """

  def __init__(self, margin: float, negative_push: float):
    """Constructs the SANLL loss."""
    super().__init__()
    assert margin > 0
    self.margin = margin
    self.negative_push = negative_push

  def get_config(self) -> dict[str, float]:
    return {
        'margin': self.margin,
        'negative_push': self.negative_push,
    }

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Computes the SANLL loss.

    Args:
      y_true: {-1.0, 1.0} float tensor holding the binary labels.
      y_pred: Tensor holding the prediction values, with same shape and type as
        y_true. Must be non-negative.

    Returns:
      A tensor with same shape and type as y_true holding non-negative,
        per-instance losses.
    """
    _assert_float_tensor(y_true)
    _assert_float_tensor(y_pred)
    valid_labels = tf.math.reduce_all(
        tf.math.logical_or(tf.equal(y_true, 1.0), tf.equal(y_true, -1.0))
    )
    check_labels = tf.debugging.Assert(
        valid_labels, [y_true], name='check_labels'
    )
    valid_scores = tf.math.reduce_all(tf.greater_equal(y_pred, 0.0))
    check_scores = tf.debugging.Assert(
        valid_scores, [y_pred], name='check_scores'
    )
    with tf.control_dependencies([check_labels, check_scores]):
      y_pred = y_pred / self.margin
      return self.margin * tf.where(
          y_true < 0,
          tf.math.pow(1.0 + y_pred, 1 + self.negative_push) - 1.0,
          -tf.math.log1p(_EPSILON - tf.math.exp(-y_pred)),
      )
