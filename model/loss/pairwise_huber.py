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
"""A smooth version of the pairwise hinge loss with a quadratic transition."""

from typing import Union

import tensorflow as tf
from tensorflow import keras


_EPSILON = 1e-6


def _assert_1d_tensor(t: tf.Tensor):
  if t.shape.rank != 1:
    raise ValueError(
        'Expected statically-known rank 1 tensor. Got shape {}'.format(t.shape)
    )


def _fast_pairwise_huber(
    soft_margin: float,
    hard_margin: float,
    pos_scores: tf.Tensor,
    neg_scores: tf.Tensor,
    pos_weights: tf.Tensor,
) -> tf.Tensor:
  """Efficient pairwise Huber loss computation, returns per-negative losses."""
  summations = tf.stack(
      [
          tf.cumsum(pos_weights),
          tf.cumsum(pos_weights * pos_scores),
          tf.cumsum(pos_weights * tf.square(pos_scores)),
      ],
      axis=1,
  )
  # Prepend 0 so we can handle maximal index from searchsorted() below.
  summations = tf.pad(
      summations, [[1, 0], [0, 0]], 'CONSTANT', constant_values=0.0
  )

  coefficients = [
      tf.stack([
          neg_scores + hard_margin,
          -tf.ones_like(neg_scores),
          tf.zeros_like(neg_scores),
      ]),
      tf.stack([
          1
          / (2 * soft_margin)
          * tf.square(neg_scores + hard_margin + soft_margin / 2),
          -1 / soft_margin * (neg_scores + hard_margin + soft_margin / 2),
          1 / (2 * soft_margin) * tf.ones_like(neg_scores),
      ]),
  ]
  boundaries = [
      None,
      neg_scores - soft_margin / 2 + hard_margin,
      neg_scores + soft_margin / 2 + hard_margin,
  ]
  assert len(coefficients) + 1 == len(boundaries)

  def contributions(bounds, coeffs):
    if bounds is None:
      return 0.0
    bix = tf.searchsorted(pos_scores, bounds)
    powers = tf.gather(summations, bix)  # shape is [#Negs, 3].
    return tf.einsum('ij,ji->i', powers, coeffs)

  losses = tf.zeros_like(neg_scores)
  for upper_bounds, lower_bounds, coeffs in zip(
      boundaries, boundaries[1:], coefficients
  ):
    excess = contributions(upper_bounds, coeffs)
    part_plus_excess = contributions(lower_bounds, coeffs)
    losses += part_plus_excess - excess

  return losses


def _fast_pairwise_hinge(
    hard_margin: float,
    pos_scores: tf.Tensor,
    neg_scores: tf.Tensor,
    pos_weights: tf.Tensor,
) -> tf.Tensor:
  """Efficient pairwise hinge loss computation, returns per-negative losses."""
  # Fold margin into positive scores.
  pos_scores = pos_scores - hard_margin
  neg_indices = tf.searchsorted(pos_scores, neg_scores, side='left')
  # Prepend 0 so we can handle maximal index from searchsorted().
  pos_scores = tf.concat([[0.0], pos_scores], axis=0)
  pos_weights = tf.concat([[0.0], pos_weights], axis=0)
  part1 = tf.gather(tf.cumsum(pos_weights), neg_indices) * neg_scores
  part2 = tf.gather(tf.cumsum(pos_scores * pos_weights), neg_indices)
  return part1 - part2


class PairwiseHuberLoss(keras.layers.Layer):
  r"""A smooth pairwise loss using a Huber-type loss function, with norm-push.

  This computes the following pairwise loss in *log-linear time*:
  \forall n \in negatives:
    loss_per_negative(n) =  \sum_{p \in positives} w(p) * l(s(n) - s(p))
  total_loss = reduce((loss_per_negative(n), w(n)) for n in negatives)

  where l() is a smooth linear-quadratic-linear non-negative loss function,
  w(x) and s(x) are the weight and score of instance x respectively, and
  reduce() is either:
    * a regular arithmetic weighted mean,
    * a weighted p-norm,
    * or a scaled and weighted log-sum-exp.
  Using the last two options will focus the optimization problem on the negative
  instances that score high.
  """

  def __init__(
      self,
      soft_margin: float,
      hard_margin: float,
      norm_push: float = 1.0,
      lse_scale: float = 0.0,
      dtype: tf.dtypes.DType = tf.float64,
  ):
    """Builds the loss layer.

    Args:
      soft_margin: the soft margin parameter controlling the shape and location
        of the transition region. Must be non-negative.
      hard_margin: the hard margin parameter controlling the overall location of
        the loss.
      norm_push: Ignored if 1, otherwise the individual negative losses are
        raised to this power before final averaging. See
        https://jmlr.csail.mit.edu/papers/v10/rudin09b.html. If not 1, then
          lse_scale must be 0.
      lse_scale: Ignored if 0, otherwise the individual negative losses are
        logsumexp reduced-ed with the given factor. As this approaches inf, the
        loss approaches the loss of the highest-scoring negative. If non-zero,
        then norm_push must be 1.
      dtype: Controls the precision of the loss computation. For very large
        input tensors (> 10M elements), tf.float32 may return wrong value such
        as negative loss or nan.
    """
    super().__init__(trainable=False, dtype=dtype)
    assert soft_margin >= 0
    assert norm_push == 1.0 or lse_scale == 0.0
    self.soft_margin = soft_margin
    self.hard_margin = hard_margin
    self.norm_push = norm_push
    self.lse_scale = lse_scale

  def get_config(self) -> dict[str, float | tf.dtypes.DType]:
    return {
        'soft_margin': self.soft_margin,
        'hard_margin': self.hard_margin,
        'norm_push': self.norm_push,
        'lse_scale': self.lse_scale,
        'dtype': self.dtype,
    }

  def call(
      self,
      inputs: Union[
          tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor]
      ],
  ) -> tf.Tensor:
    """Computes the pairwise Huber loss efficiently.

    Args:
      inputs: Either a pair or a 3-element tuple of rank 1 tensors, each
        respectively representing the binary labels, instance scores and
        instance weights. The labels are a float tensor such that values greater
        than 0.5 represent positive labels. All tensors must have the same size.

    Returns:
      loss: a scalar non-negative tensor holding the loss.
    """
    labels = inputs[0]
    scores = inputs[1]
    weights = inputs[2] if len(inputs) > 2 else None
    _assert_1d_tensor(scores)
    _assert_1d_tensor(labels)

    with tf.control_dependencies(
        [tf.assert_equal(tf.shape(labels), tf.shape(scores))]
    ):
      pos_indicator = labels > 0.5
      neg_scores = tf.boolean_mask(scores, tf.logical_not(pos_indicator))
      pos_scores = tf.boolean_mask(scores, pos_indicator)

    ix = tf.argsort(pos_scores)
    pos_scores = tf.gather(pos_scores, ix)
    zero_constant = tf.constant(0.0, dtype=self.dtype)

    if weights is None:
      pos_weights = tf.ones_like(pos_scores)
      neg_weights = tf.ones_like(neg_scores)
    else:
      _assert_1d_tensor(weights)
      with tf.control_dependencies([
          tf.debugging.assert_equal(tf.shape(labels), tf.shape(weights)),
          tf.debugging.assert_greater_equal(weights, zero_constant),
      ]):
        pos_weights = tf.gather(tf.boolean_mask(weights, pos_indicator), ix)
      neg_weights = tf.boolean_mask(weights, tf.logical_not(pos_indicator))
    pos_weights /= tf.reduce_sum(pos_weights)
    neg_weights /= tf.reduce_sum(neg_weights)

    if self.soft_margin > 0.0:
      losses = _fast_pairwise_huber(
          self.soft_margin,
          self.hard_margin,
          pos_scores,
          neg_scores,
          pos_weights,
      )
    else:
      losses = _fast_pairwise_hinge(
          self.hard_margin, pos_scores, neg_scores, pos_weights
      )

    if self.lse_scale != 0.0:
      losses = self.lse_scale * losses + tf.math.log(neg_weights)
      loss = 1 / self.lse_scale * tf.math.reduce_logsumexp(losses)
      # Handle no negatives: reduce_logsumexp returns -inf on empty inputs.
      return tf.cond(
          tf.math.is_finite(loss),
          lambda: loss,
          lambda: tf.constant(0.0, dtype=loss.dtype),
      )

    if self.norm_push != 1.0:
      losses = tf.math.pow(losses + _EPSILON, self.norm_push)
    loss = tf.einsum('i,i->', losses, neg_weights)
    if self.norm_push != 1.0:
      return tf.math.pow(loss + _EPSILON, 1 / self.norm_push)
    return loss


class PairwiseHuberKerasLoss(keras.losses.Loss):
  """See PairwiseHuberLoss."""

  def __init__(
      self,
      soft_margin: float,
      hard_margin: float,
      norm_push: float = 1.0,
      lse_scale: float = 0.0,
  ):
    super().__init__()
    self._loss = PairwiseHuberLoss(
        soft_margin, hard_margin, norm_push, lse_scale
    )

  def call(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      sample_weight: tf.Tensor | None = None,
  ) -> tf.Tensor:
    if sample_weight is not None:
      return self._loss((y_true, y_pred))
    else:
      return self._loss((y_true, y_pred, sample_weight))
