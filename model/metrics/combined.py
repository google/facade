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
"""Exact in-memory metrics for large-scale datasets."""

from typing import Any, Optional

import tensorflow as tf
from tensorflow import keras

from model.metrics import concatenator


def _normalized_cumsum(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
  total = tf.math.reduce_sum(x)
  zero = tf.constant(0, dtype=x.dtype)
  x = tf.pad(x, [[1, 0]], 'CONSTANT', constant_values=zero)
  x = tf.math.cumsum(x)
  x = tf.math.divide_no_nan(x, total)
  return x, total


def _less_or_equal(keys: tf.Tensor, queries: list[float]) -> tf.Tensor:
  """For every q in queries, largest index i such that keys[i] <= q."""
  queries = tf.constant(queries, dtype=keys.dtype)
  indices = tf.searchsorted(
      keys, queries, side='right', out_type=tf.dtypes.int64
  )
  indices = indices - 1
  return indices


def _auc_at_fprs(
    fprs: tf.Tensor, tprs: tf.Tensor, fpr_thresholds: list[float]
) -> list[tf.Tensor]:
  """Computes partial ROC-AUC, assuming all predictions are distinct."""
  indices = _less_or_equal(fprs, fpr_thresholds)
  aucs = []
  for i in range(len(fpr_thresholds)):
    b = indices[i]
    norm_factor = fprs[b]  # max achievable partial AUC.
    delta_fpr = fprs[1 : b + 1] - fprs[:b]
    average_tpr = 0.5 * (tprs[1 : b + 1] + tprs[:b])
    # The ROC curve is piecewise linear, so this summation gives the exact area.
    raw_auc = tf.einsum('i,i->', delta_fpr, average_tpr)
    aucs.append(tf.math.divide_no_nan(raw_auc, norm_factor))
  return aucs


def _dyn_boolean_mask(x: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
  """Like tf.boolean_mask, but allows for unknown tensor shapes."""
  return tf.gather_nd(x, tf.where(mask))


def _normalized_mean_log_tail_probability(
    predictions: tf.Tensor, labels: tf.Tensor, weights: tf.Tensor
) -> tf.Tensor:
  """Computes the (un)normalized mean log tail probability metrics."""
  # Compute the Empirical Tail Distribution Function (ETDF) of scores of
  # positive-weight negative instances.
  neg_labels = tf.math.logical_and(tf.math.logical_not(labels), weights > 0)
  negative_predictions = _dyn_boolean_mask(predictions, neg_labels)
  negative_weights = _dyn_boolean_mask(weights, neg_labels)
  # Start the summation from the right to minimize loss of precision where it
  # matters.
  sorting_indices = tf.argsort(negative_predictions, direction='DESCENDING')
  negative_predictions = tf.gather(negative_predictions, sorting_indices)
  negative_weights = tf.gather(negative_weights, sorting_indices)

  def non_empty_negatives():
    nonlocal negative_weights, negative_predictions
    # Repeat the highest scoring observation to avoid collapsing the TDF
    # to zero at any score threshold.
    negative_weights = tf.pad(negative_weights, [[1, 0]], 'SYMMETRIC')
    negative_tdf, _ = _normalized_cumsum(negative_weights)
    # Re-arange so that thresholds are sorted by increasing order.
    negative_tdf = tf.reverse(negative_tdf, [0])
    negative_predictions = tf.reverse(negative_predictions, [0])

    positive_predictions = _dyn_boolean_mask(predictions, labels)
    # For every positive prediction s, smallest index i such that
    # negative_predictions[i] >= s.
    ix = tf.searchsorted(
        negative_predictions,
        positive_predictions,
        out_type=tf.dtypes.int64,
    )
    tail_ps = tf.gather(negative_tdf, ix)
    log_tail_ps = tf.math.log(tail_ps)
    positive_weights = _dyn_boolean_mask(weights, labels)
    mean_log_tail_p = tf.einsum(
        'i,i->', log_tail_ps, positive_weights
    ) / tf.reduce_sum(positive_weights)
    best_achievable = tf.math.log(negative_tdf[-2])

    mean_log10_tail_p = -mean_log_tail_p / tf.math.log(
        tf.constant(10.0, dtype=mean_log_tail_p.dtype)
    )

    return mean_log10_tail_p, mean_log_tail_p / best_achievable

  # Weights are float64, so non_empty_negatives returns float64 too.
  nan = tf.constant(float('nan'), dtype=tf.float64)
  return tf.cond(
      tf.math.greater(tf.size(negative_predictions), 0),
      non_empty_negatives,
      lambda: (nan, nan),
  )


def _assert_1d_tensor(t: tf.Tensor):
  if t.shape.rank != 1:
    raise ValueError(
        f'Expected statically-known rank 1 tensor. Got shape {t.shape}.'
    )


def _assert_non_negative(x: Optional[list[float]]):
  if not x:
    return
  for value in x:
    if value < 0:
      raise ValueError(f'Expected non-negative float, got {value}.')


def _compute_metrics(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    weights: tf.Tensor,
    tpr_at_fpr_thresholds: Optional[list[float]] = None,
    auc_at_fpr_thresholds: Optional[list[float]] = None,
    prediction_at_fpr_thresholds: Optional[list[float]] = None,
    name: Optional[str] = None,
) -> dict[str, tf.Tensor]:
  """Computes all metrics of interest given labels, predictions and weights."""
  prefix = f'{name}/' if name else ''
  metric_values = {}

  # Normalized mean log tail probability metric.
  mltp, nmltp = _normalized_mean_log_tail_probability(
      predictions, labels, weights
  )
  metric_values[f'{prefix}mean_log10_tail_probability'] = mltp
  metric_values[f'{prefix}normalized_mean_log_tail_probability'] = nmltp

  # ROC-curve based metrics.
  sorting_indices = tf.argsort(predictions, direction='DESCENDING')
  predictions = tf.gather(predictions, sorting_indices)
  labels = tf.gather(labels, sorting_indices)
  weights = tf.gather(weights, sorting_indices)

  zero = tf.constant(0, dtype=weights.dtype)
  tprs, total_pos_weight = _normalized_cumsum(tf.where(labels, weights, zero))
  fprs, total_neg_weight = _normalized_cumsum(tf.where(labels, zero, weights))
  nan = tf.constant(float('nan'), dtype=predictions.dtype)
  predictions = tf.pad(predictions, [[1, 0]], 'CONSTANT', constant_values=nan)

  if tpr_at_fpr_thresholds:
    values = tf.gather(tprs, _less_or_equal(fprs, tpr_at_fpr_thresholds))
    for i, threshold in enumerate(tpr_at_fpr_thresholds):
      metric_values[f'{prefix}tpr_at_fpr_{threshold:.1e}'] = values[i]

  if auc_at_fpr_thresholds:
    values = _auc_at_fprs(fprs, tprs, auc_at_fpr_thresholds)
    for i, threshold in enumerate(auc_at_fpr_thresholds):
      metric_values[f'{prefix}auc_at_fpr_{threshold:.1e}'] = values[i]

  if prediction_at_fpr_thresholds:
    values = tf.gather(
        predictions, _less_or_equal(fprs, prediction_at_fpr_thresholds)
    )
    for i, threshold in enumerate(prediction_at_fpr_thresholds):
      metric_values[f'{prefix}score_at_fpr_{threshold:.1e}'] = values[i]

  # Optimal weighted error metric.
  errors_at_thresholds = total_neg_weight * fprs - total_pos_weight * tprs
  argmin = tf.math.argmin(errors_at_thresholds, output_type=tf.dtypes.int64)
  min_error = (total_pos_weight + errors_at_thresholds[argmin]) / (
      total_pos_weight + total_neg_weight
  )
  metric_values[f'{prefix}best_weighted_error'] = min_error
  metric_values[f'{prefix}score_for_best_weighted_error'] = predictions[argmin]
  return metric_values


class ExactTruncatedMetrics(keras.metrics.Metric):
  """Computes various metrics parametrized by false positive rate exactly.

  This constructs a subgraph which accumulates in local memory labels,
  predictions and weights across minibatches, and uses those to compute the
  following metrics exactly:
    * true positive rates at the specified false positive rates,
    * partial ROC-AUCs at the specified false positive rates,
    * prediction scores at the specified false positive rates,
    * best achievable weighted error rate and its corresponding score threshold,
    * unnormalized and normalized mean log tail probability (MLTP) of the
      positives in the distribution of negative instances.

  The weighted error rate at a given score threshold is defined as the sum of
  false positive and false negative instance weights, divided by the total
  weight across all instances alike.

  The Mean Log Tail Probability (MLTP) metric is the opposite of the mean of the
  log-base-10 tail probabilities of negative examples at every positive example
  score. The normalized MLTP is the MTLP divided by the best achievable MLTP
  (when every positive example is above the highest scoring negative).
  """

  def __init__(
      self,
      tpr_at_fpr_thresholds: Optional[list[float]] = None,
      auc_at_fpr_thresholds: Optional[list[float]] = None,
      prediction_at_fpr_thresholds: Optional[list[float]] = None,
      name_prefix: Optional[str] = None,
  ):
    """Constructs the Metric.

    Args:
      tpr_at_fpr_thresholds: for every non-negative float in the list, output
        dictionary will have a corresponding entry tracking the best achievable
        true positive rate such that the false positive rate is less or equal to
        the specified threshold.
      auc_at_fpr_thresholds: for every non-negative float in the list, output
        dictionary will have a corresponding entry tracking the best achievable
        ROC-AUC such that the false positive rate is less or equal to the
        specified threshold. This is also known as the truncated AUC@FPR. Use
        any threshold >= 1.0 to measure the full ROC-AUC.
      prediction_at_fpr_thresholds: for every non-negative float in the list,
        output dictionary will have a corresponding entry tracking prediction
        score at or below the specified false positive rate threshold.
      name_prefix: Optional common prefix for all sub-metric results.

    Raises:
      ValueError: if any false positive rate threshold is negative.
    """
    super().__init__()
    _assert_non_negative(tpr_at_fpr_thresholds)
    _assert_non_negative(auc_at_fpr_thresholds)
    _assert_non_negative(prediction_at_fpr_thresholds)
    self._tpr_at_fpr_thresholds = tpr_at_fpr_thresholds
    self._auc_at_fpr_thresholds = auc_at_fpr_thresholds
    self._prediction_at_fpr_thresholds = prediction_at_fpr_thresholds
    self._labels = concatenator.Concatenator(dtype=tf.bool)
    self._predictions = concatenator.Concatenator(dtype=tf.float32)
    # Use high precision for weights given large summations.
    self._weights = concatenator.Concatenator(dtype=tf.float64)
    self._name_prefix = name_prefix

  def get_config(self) -> dict[str, Any]:
    return {
        'tpr_at_fpr_thresholds': self._tpr_at_fpr_thresholds,
        'auc_at_fpr_thresholds': self._auc_at_fpr_thresholds,
        'prediction_at_fpr_thresholds': self._prediction_at_fpr_thresholds,
        'name_prefix': self._name_prefix,
    }

  def reset_state(self):
    self._labels.reset_state()
    self._predictions.reset_state()
    self._weights.reset_state()

  @tf.function(reduce_retracing=True)
  def update_state(
      self,
      labels: tf.Tensor,
      predictions: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ):
    """Update state by storing inputs.

    This constructs a subgraph which accumulates in local memory labels,
    predictions and weights across minibatches.

    Args:
      labels: rank-1 float tensor of such that values greater than 0.5 represent
        the positive labels.
      predictions: rank-1 tensor of scores. Must have same shape as `labels`.
      sample_weight: rank-1 tensor of instance weights. Must have shape as
        `labels`.

    Raises:
      ValueError: if input tensors are not rank 1.
    """
    _assert_1d_tensor(labels)
    _assert_1d_tensor(predictions)
    if sample_weight is None:
      sample_weight = tf.ones_like(predictions, tf.float64)
    _assert_1d_tensor(sample_weight)

    with tf.control_dependencies([
        tf.assert_equal(tf.shape(labels), tf.shape(predictions)),
        tf.assert_equal(tf.shape(labels), tf.shape(sample_weight)),
    ]):
      labels = labels > 0.5
      # (Possibly) downcast scores as those are only used for sorting.
      predictions = tf.cast(predictions, tf.float32)
      # (Possibly) upcast weights as those are summed together.
      sample_weight = tf.cast(sample_weight, tf.float64)

    self._labels.update_state(labels)
    self._predictions.update_state(predictions)
    self._weights.update_state(sample_weight)

  def result(self) -> dict[str, tf.Tensor]:
    """Computes requested metrics from stored data.

    Returns:
      metric_values: a dictionary of metric name (str) to metric value (a scalar
      tensor).
    """
    return _compute_metrics(
        self._labels.result(),
        self._predictions.result(),
        self._weights.result(),
        self._tpr_at_fpr_thresholds,
        self._auc_at_fpr_thresholds,
        self._prediction_at_fpr_thresholds,
        self._name_prefix,
    )
