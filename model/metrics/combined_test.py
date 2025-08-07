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
import bisect
import inspect
import math

import numpy as np
import tensorflow as tf

from model.metrics import combined
from parameterized import parameterized


def masked_normalized_cumsum(mask: np.ndarray, xs: np.ndarray) -> np.ndarray:
  result = np.cumsum(np.concatenate([[0], np.where(mask, xs, 0.0)], axis=0))
  if result[-1] > 0:
    result /= result[-1]
  return result


def roc_curve(
    labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  sorting_indices = np.argsort(predictions)[::-1]
  predictions = predictions[sorting_indices]
  weights = weights[sorting_indices]
  labels = labels[sorting_indices]
  tprs = masked_normalized_cumsum(labels > 0.5, weights)
  fprs = masked_normalized_cumsum(labels < 0.5, weights)
  predictions = np.concatenate([[float('nan')], predictions], axis=0)
  return fprs, tprs, predictions


def area_under_curve(x: np.ndarray, y: np.ndarray) -> float:
  return np.sum((x[1:] - x[:-1]) * 0.5 * (y[:-1] + y[1:]))


def best_weighted_error_and_score(
    labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
  sorting_indices = np.argsort(predictions)
  predictions = predictions[sorting_indices]
  labels = labels[sorting_indices]
  weights = weights[sorting_indices]
  false_negative_weight = np.cumsum(
      np.concatenate([[0], np.where(labels > 0.5, weights, 0.0)])
  )
  false_positive_weight = np.cumsum(
      np.concatenate([np.where(labels < 0.5, weights, 0.0), [0]])[::-1]
  )[::-1]
  errors = false_negative_weight + false_positive_weight
  i = np.argmin(errors)
  score = float('nan')
  if i < predictions.shape[0]:
    score = predictions[i]
  total_pos_weight = np.sum(weights[labels > 0.5])
  total_neg_weight = np.sum(weights[labels < 0.5])
  return errors[i] / (total_pos_weight + total_neg_weight), score


def normalized_mean_log_tail_probability(
    labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> float:
  negatives = np.logical_and(labels < 0.5, weights > 0)
  negative_scores = predictions[negatives]
  negative_weights = weights[negatives]
  ix = np.argsort(negative_scores)
  negative_scores = negative_scores[ix]
  negative_weights = negative_weights[ix]
  negative_weights = np.concatenate(
      [[0.0], negative_weights, [negative_weights[-1]]]
  )
  negative_weights /= np.sum(negative_weights)
  tdf = 1.0 - np.cumsum(negative_weights)

  positives = labels > 0.5
  positive_scores = predictions[positives]
  positive_weights = weights[positives]
  ix = np.searchsorted(negative_scores, positive_scores)
  mean_log_tail_p = np.sum(positive_weights * np.log10(tdf[ix])) / np.sum(
      positive_weights
  )
  best_log_tail_p = np.log10(tdf[-2])
  return -mean_log_tail_p, mean_log_tail_p / best_log_tail_p


class ExactTruncatedMetricsTest(tf.test.TestCase):

  def test_accepts_empty_predictions(self):
    metrics = combined.ExactTruncatedMetrics(
        tpr_at_fpr_thresholds=[0.5],
        auc_at_fpr_thresholds=[0.5],
        prediction_at_fpr_thresholds=[0.5],
    )

    metrics.update_state(
        labels=tf.constant(np.empty([0])),
        predictions=tf.constant(np.empty([0])),
        sample_weight=tf.constant(np.empty([0])),
    )

    metric_name_value = metrics.result()

    self.assertAllEqual(metric_name_value['auc_at_fpr_5.0e-01'], 0.0)
    self.assertAllEqual(metric_name_value['tpr_at_fpr_5.0e-01'], 0.0)
    self.assertTrue(math.isnan(metric_name_value['score_at_fpr_5.0e-01']))
    self.assertTrue(
        math.isnan(metric_name_value['normalized_mean_log_tail_probability'])
    )

  def test_applies_name_prefix(self):
    metrics = combined.ExactTruncatedMetrics(
        tpr_at_fpr_thresholds=[0.5],
        auc_at_fpr_thresholds=[0.5],
        prediction_at_fpr_thresholds=[0.5],
        name_prefix='prefix',
    )
    metric_name_value = metrics.result()
    self.assertAllEqual(metric_name_value['prefix/auc_at_fpr_5.0e-01'], 0.0)
    self.assertAllEqual(metric_name_value['prefix/tpr_at_fpr_5.0e-01'], 0.0)
    self.assertTrue(
        math.isnan(metric_name_value['prefix/score_at_fpr_5.0e-01'])
    )
    self.assertTrue(
        math.isnan(
            metric_name_value['prefix/normalized_mean_log_tail_probability']
        )
    )

  def test_simple_case(self):
    fprs_thresholds = [0.0, 0.2, 0.25, 0.30, 0.5, 1.0, 1.1]
    metrics = combined.ExactTruncatedMetrics(
        tpr_at_fpr_thresholds=fprs_thresholds,
        auc_at_fpr_thresholds=fprs_thresholds,
        prediction_at_fpr_thresholds=fprs_thresholds,
    )

    all_labels = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    all_predictions = [-0.1, 0.0, -0.2, 0.4, 0.1, 0.3, 0.5, 0.2]
    all_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # predictions: .5  .4  .3  .2  .1  .0  -.1 -.2
    # labels:      +   -   +   +   +   -   -   -
    # tprs:        .25 .25 .50 .75 1.0 1.0 1.0 1.0
    # fprs:        0.0 .25 .25 .25 .25 .50 .75 1.0
    # fps:         0   1   1   1   1   2   3   4
    # fns:         3   3   2   1   0   0   0   0
    metrics.update_state(
        labels=tf.constant(all_labels),
        predictions=tf.constant(all_predictions),
        sample_weight=tf.constant(all_weights),
    )
    metric_name_value = metrics.result()

    self.assertAllClose(metric_name_value['tpr_at_fpr_0.0e+00'], 0.25)
    self.assertAllClose(metric_name_value['auc_at_fpr_0.0e+00'], 0.0)
    self.assertAllClose(metric_name_value['score_at_fpr_0.0e+00'], 0.5)

    self.assertAllClose(metric_name_value['tpr_at_fpr_2.0e-01'], 0.25)
    self.assertAllClose(metric_name_value['auc_at_fpr_2.0e-01'], 0.0)
    self.assertAllClose(metric_name_value['score_at_fpr_2.0e-01'], 0.5)

    self.assertAllClose(metric_name_value['tpr_at_fpr_2.5e-01'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_2.5e-01'], 0.25)
    self.assertAllClose(metric_name_value['score_at_fpr_2.5e-01'], 0.1)

    self.assertAllClose(metric_name_value['tpr_at_fpr_3.0e-01'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_3.0e-01'], 0.25)
    self.assertAllClose(metric_name_value['score_at_fpr_3.0e-01'], 0.1)

    self.assertAllClose(metric_name_value['tpr_at_fpr_5.0e-01'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_5.0e-01'], (0.25 + 1) / 2)
    self.assertAllClose(metric_name_value['score_at_fpr_5.0e-01'], 0.0)

    self.assertAllClose(metric_name_value['tpr_at_fpr_1.0e+00'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_1.0e+00'], 1 / 16 + 3 / 4)
    self.assertAllClose(metric_name_value['score_at_fpr_1.0e+00'], -0.2)

    self.assertAllClose(metric_name_value['tpr_at_fpr_1.1e+00'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_1.1e+00'], 1 / 16 + 3 / 4)
    self.assertAllClose(metric_name_value['score_at_fpr_1.1e+00'], -0.2)

    self.assertAllClose(metric_name_value['best_weighted_error'], 1.0 / 8)
    self.assertAllClose(metric_name_value['score_for_best_weighted_error'], 0.1)

    # Empirical Tail Distribution Function of negative examples
    # x:                          -.2   -.1  0.0   .4   \infty
    # tail_p(x):                  5/5   4/5  3/5   2/5  1/5
    # tail_ps @ positive scores: .5 ->  1/5
    #                            .3 ->  2/5
    #                            .2 ->  2/5
    #                            .1 ->  2/5
    mltp = -(np.log10(1 / 5) + 3 * np.log10(2 / 5)) / 4
    nmltp = -mltp / np.log10(1 / 5)
    # Sanity checks.
    self.assertGreater(mltp, 0.0)
    self.assertGreater(nmltp, 0.0)
    self.assertLess(nmltp, 1.0)
    self.assertAllClose(metric_name_value['mean_log10_tail_probability'], mltp)
    self.assertAllClose(
        metric_name_value['normalized_mean_log_tail_probability'], nmltp
    )

  def test_simple_case_default_weight(self):
    fprs_thresholds = [0.0, 0.2, 0.25, 0.30, 0.5, 1.0, 1.1]
    metrics = combined.ExactTruncatedMetrics(
        tpr_at_fpr_thresholds=fprs_thresholds,
        auc_at_fpr_thresholds=fprs_thresholds,
        prediction_at_fpr_thresholds=fprs_thresholds,
    )

    # Same values as above.
    all_labels = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    all_predictions = [-0.1, 0.0, -0.2, 0.4, 0.1, 0.3, 0.5, 0.2]
    metrics.update_state(
        labels=tf.constant(all_labels),
        predictions=tf.constant(all_predictions),
    )
    metric_name_value = metrics.result()

    self.assertAllClose(metric_name_value['tpr_at_fpr_0.0e+00'], 0.25)
    self.assertAllClose(metric_name_value['auc_at_fpr_0.0e+00'], 0.0)
    self.assertAllClose(metric_name_value['score_at_fpr_0.0e+00'], 0.5)

    self.assertAllClose(metric_name_value['tpr_at_fpr_2.0e-01'], 0.25)
    self.assertAllClose(metric_name_value['auc_at_fpr_2.0e-01'], 0.0)
    self.assertAllClose(metric_name_value['score_at_fpr_2.0e-01'], 0.5)

    self.assertAllClose(metric_name_value['tpr_at_fpr_2.5e-01'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_2.5e-01'], 0.25)
    self.assertAllClose(metric_name_value['score_at_fpr_2.5e-01'], 0.1)

    self.assertAllClose(metric_name_value['tpr_at_fpr_3.0e-01'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_3.0e-01'], 0.25)
    self.assertAllClose(metric_name_value['score_at_fpr_3.0e-01'], 0.1)

    self.assertAllClose(metric_name_value['tpr_at_fpr_5.0e-01'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_5.0e-01'], (0.25 + 1) / 2)
    self.assertAllClose(metric_name_value['score_at_fpr_5.0e-01'], 0.0)

    self.assertAllClose(metric_name_value['tpr_at_fpr_1.0e+00'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_1.0e+00'], 1 / 16 + 3 / 4)
    self.assertAllClose(metric_name_value['score_at_fpr_1.0e+00'], -0.2)

    self.assertAllClose(metric_name_value['tpr_at_fpr_1.1e+00'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_1.1e+00'], 1 / 16 + 3 / 4)
    self.assertAllClose(metric_name_value['score_at_fpr_1.1e+00'], -0.2)

    self.assertAllClose(metric_name_value['best_weighted_error'], 1.0 / 8)
    self.assertAllClose(metric_name_value['score_for_best_weighted_error'], 0.1)

    mltp = -(np.log10(1 / 5) + 3 * np.log10(2 / 5)) / 4
    nmltp = -mltp / np.log10(1 / 5)
    # Sanity checks.
    self.assertGreater(mltp, 0.0)
    self.assertGreater(nmltp, 0.0)
    self.assertLess(nmltp, 1.0)
    self.assertAllClose(metric_name_value['mean_log10_tail_probability'], mltp)
    self.assertAllClose(
        metric_name_value['normalized_mean_log_tail_probability'], nmltp
    )

  def test_resets(self):
    metrics = combined.ExactTruncatedMetrics(
        auc_at_fpr_thresholds=[1.0],
    )

    all_labels = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    all_predictions = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2]
    all_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    metrics.update_state(
        labels=tf.constant(all_labels),
        predictions=tf.constant(all_predictions),
        sample_weight=tf.constant(all_weights),
    )
    metric_name_value = metrics.result()
    self.assertAllClose(metric_name_value['auc_at_fpr_1.0e+00'], 0.0)
    metrics.reset_state()

    all_labels = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    metrics.update_state(
        labels=tf.constant(all_labels),
        predictions=tf.constant(all_predictions),
        sample_weight=tf.constant(all_weights),
    )
    metric_name_value = metrics.result()
    self.assertAllClose(metric_name_value['auc_at_fpr_1.0e+00'], 1.0)

  def test_streaming_accumulates(self):
    fprs_thresholds = [0.5]
    metrics = combined.ExactTruncatedMetrics(
        tpr_at_fpr_thresholds=fprs_thresholds,
        auc_at_fpr_thresholds=fprs_thresholds,
        prediction_at_fpr_thresholds=fprs_thresholds,
    )

    all_labels = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    all_predictions = [-0.1, 0.0, -0.2, 0.4, 0.1, 0.3, 0.5, 0.2]
    all_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # predictions: .5  .4  .3  .2  .1  .0  -.1 -.2
    # labels:      +   -   +   +   +   -   -   -
    # tprs:        .25 .25 .50 .75 1.0 1.0 1.0 1.0
    # fprs:        0.0 .25 .25 .25 .25 .50 .75 1.0
    metrics.update_state(
        labels=tf.constant(all_labels[:3]),
        predictions=tf.constant(all_predictions[:3]),
        sample_weight=tf.constant(all_weights[:3]),
    )
    metrics.update_state(
        labels=tf.constant(all_labels[3:5]),
        predictions=tf.constant(all_predictions[3:5]),
        sample_weight=tf.constant(all_weights[3:5]),
    )
    metrics.update_state(
        labels=tf.constant(all_labels[5:]),
        predictions=tf.constant(all_predictions[5:]),
        sample_weight=tf.constant(all_weights[5:]),
    )
    metric_name_value = metrics.result()

    self.assertAllClose(metric_name_value['tpr_at_fpr_5.0e-01'], 1.0)
    self.assertAllClose(metric_name_value['auc_at_fpr_5.0e-01'], (0.25 + 1) / 2)
    self.assertAllClose(metric_name_value['score_at_fpr_5.0e-01'], 0.0)

  @parameterized.expand([
      [1, 50, 50, 0.01],
      [2, 50, 50, 0.5],
      [3, 50, 50, 1.1],
      [4, 1000, 1, 0.01],
      [5, 1000, 1, 0.5],
      [6, 1000, 1, 1.1],
      [7, 1000, 1000, 0.01],
      [8, 1000, 1000, 0.1],
      [9, 1000, 1000, 1.1],
      [10, 100000, 100000, 0.001],
      [11, 100000, 100000, 0.7],
      [12, 100000, 100000, 1.1],
  ])
  def test_large_scale_correctness(
      self, rng_seed, n_negatives, n_positives, fpr
  ):
    rng = np.random.default_rng(rng_seed)
    negatives = rng.normal(size=n_negatives)
    positives = rng.normal(size=n_positives)
    all_labels = np.concatenate(
        [np.zeros_like(negatives), np.ones_like(positives)], axis=0
    )
    all_predictions = np.concatenate([negatives, positives], axis=0)
    all_weights = rng.uniform(size=n_negatives + n_positives).astype(np.float64)

    metrics = combined.ExactTruncatedMetrics(
        tpr_at_fpr_thresholds=[fpr],
        auc_at_fpr_thresholds=[fpr],
        prediction_at_fpr_thresholds=[fpr],
    )

    metrics.update_state(
        labels=tf.constant(all_labels),
        predictions=tf.constant(all_predictions),
        sample_weight=tf.constant(all_weights),
    )
    metric_name_value = metrics.result()

    fprs, tprs, scores = roc_curve(all_labels, all_predictions, all_weights)
    index = bisect.bisect_right(fprs, fpr) - 1
    epsilon = 1e-12
    auc = area_under_curve(fprs[: index + 1], tprs[: index + 1]) / (
        fprs[index] + epsilon
    )
    tpr = tprs[index]
    score = scores[index]
    self.assertAllClose(metric_name_value[f'tpr_at_fpr_{fpr:.1e}'], tpr)
    self.assertAllClose(metric_name_value[f'auc_at_fpr_{fpr:.1e}'], auc)
    self.assertAllClose(metric_name_value[f'score_at_fpr_{fpr:.1e}'], score)

    best_error, best_score = best_weighted_error_and_score(
        all_labels, all_predictions, all_weights
    )
    self.assertAllClose(metric_name_value['best_weighted_error'], best_error)
    self.assertAllClose(
        metric_name_value['score_for_best_weighted_error'], best_score
    )

    mltp, nmltp = normalized_mean_log_tail_probability(
        all_labels, all_predictions, all_weights
    )
    self.assertAllClose(metric_name_value['mean_log10_tail_probability'], mltp)
    self.assertAllClose(
        metric_name_value['normalized_mean_log_tail_probability'], nmltp
    )

  def test_serializes(self):
    config = {
        'tpr_at_fpr_thresholds': [0.2],
        'auc_at_fpr_thresholds': [0.4],
        'prediction_at_fpr_thresholds': [0.8],
        'name_prefix': None,
    }
    f = combined.ExactTruncatedMetrics.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(combined.ExactTruncatedMetrics).parameters.keys(),
        config.keys(),
    )


if __name__ == '__main__':
  tf.test.main()
