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
"""Logistic loss with configurable margins."""

import tensorflow as tf
from tensorflow import keras


def _assert_float_tensor(t: tf.Tensor):
  if t.dtype not in [tf.float32, tf.float64]:
    raise ValueError('Expected float-type tensor. Got: {}'.format(t))


def _hinge(
    y_true: tf.Tensor, y_pred: tf.Tensor, hard_margin: float
) -> tf.Tensor:
  return tf.math.maximum(hard_margin - y_true * y_pred, 0.0)


def _logistic(
    y_true: tf.Tensor, y_pred: tf.Tensor, soft_margin: float, hard_margin: float
) -> tf.Tensor:
  return soft_margin * tf.math.softplus(
      (hard_margin - y_true * y_pred) / soft_margin
  )


def _negative_push(
    losses: tf.Tensor, y_true: tf.Tensor, negative_push: float
) -> tf.Tensor:
  if negative_push == 0.0:
    return losses
  return tf.math.pow(1 + losses, 1 + (1.0 - y_true) * negative_push / 2.0) - 1


class GeneralizedLogisticLoss(keras.losses.Loss):
  r"""Generalized logistic loss: configurable soft and hard margins.

  For an instance with {-1, +1} label y and score x, the generalized logistic
  loss is:
    loss(y, x) = (1 + s * softplus((h - y*x) / s))^(1 + (1-y)/2 * p) - 1, where:
    * s is the soft_margin parameter,
    * h is the hard_margin parameter.
    * p is the negative_push parameter.

  When p=0 and s->0, the loss collapses to a hinge loss with configurable (hard)
    margin: loss(y, x) = max(0, h - y*x).

  When p=0, the loss is symmetric for negative and positive instances. p does
  not influence the loss on the positive instances. Using p > 0 increases the
  loss of negatives asymptotically as y^p, which is useful for penalizing
  high-scoring negatives.
  """

  def __init__(
      self, soft_margin: float, hard_margin: float, negative_push: float
  ):
    """Builds the loss."""
    super().__init__()
    assert soft_margin >= 0
    assert hard_margin >= 0
    self.soft_margin = soft_margin
    self.hard_margin = hard_margin
    self.negative_push = negative_push

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Computes the loss.

    Args:
      y_true: {-1.0, 1.0} float tensor holding the binary labels.
      y_pred: Tensor holding the prediction values, with same shape and type as
        y_true.

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
    with tf.control_dependencies([check_labels]):
      if self.soft_margin == 0.0:
        return _negative_push(
            _hinge(y_true, y_pred, self.hard_margin), y_true, self.negative_push
        )
      return _negative_push(
          _logistic(y_true, y_pred, self.soft_margin, self.hard_margin),
          y_true,
          self.negative_push,
      )

  def get_config(self) -> dict[str, float]:
    return {
        'soft_margin': self.soft_margin,
        'hard_margin': self.hard_margin,
        'negative_push': self.negative_push,
    }
