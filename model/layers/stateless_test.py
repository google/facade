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
import inspect

import numpy as np
import tensorflow as tf

from model.layers import stateless
from protos import config_pb2


TR = config_pb2.Transformation
SF = config_pb2.ScoringFunction


class EmbeddingsTransformationTest(tf.test.TestCase):

  def test_chaining_works(self):
    f = stateless.EmbeddingsTransformation([
        TR.TR_SOFTPLUS,
        TR.TR_L2_NORMALIZED,
    ])
    cases = np.array([[0, 0], [-1, 1], [1, 1]], dtype=np.float32)
    x = tf.constant(cases)
    result = f(x)
    expected = np.log1p(np.exp(cases))
    expected /= np.linalg.norm(expected, ord=2, axis=1, keepdims=True)
    self.assertAllClose(result, expected)

  def test_all_options_work(self):
    cases = np.array([[0, 0], [-1, 1], [1, 1]], dtype=np.float32)
    x = tf.constant(cases)
    identity = stateless.EmbeddingsTransformation([TR.TR_IDENTITY])(x)
    sigmoid = stateless.EmbeddingsTransformation([TR.TR_SIGMOID])(x)
    softplus = stateless.EmbeddingsTransformation([TR.TR_SOFTPLUS])(x)
    softmax = stateless.EmbeddingsTransformation([TR.TR_SOFTMAX])(x)
    l2_normalized = stateless.EmbeddingsTransformation([TR.TR_L2_NORMALIZED])(x)

    self.assertAllClose(identity, cases)
    self.assertAllClose(sigmoid, 1 / (1 + np.exp(-cases)))
    self.assertAllClose(softplus, np.log1p(np.exp(cases)))
    self.assertAllClose(
        softmax, np.exp(cases) / np.sum(np.exp(cases), axis=1, keepdims=True)
    )
    epsilon = 1e-6
    self.assertAllClose(
        l2_normalized,
        cases / (np.linalg.norm(cases, ord=2, axis=1, keepdims=True) + epsilon),
    )

  def test_handles_all_enum_values(self):
    cases = np.array([[0, 0], [-1, 1], [1, 1]], dtype=np.float32)
    x = tf.constant(cases)
    for transformation in TR.values():
      if transformation == TR.TR_UNSPECIFIED:
        continue
      y = stateless.EmbeddingsTransformation([transformation])(x)
      self.assertNotEmpty(y)

  def test_serializes(self):
    config = {
        'transformations': [
            TR.TR_SOFTPLUS,
            TR.TR_L2_NORMALIZED,
        ]
    }
    f = stateless.EmbeddingsTransformation.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(stateless.EmbeddingsTransformation).parameters.keys(),
        config.keys(),
    )


class EmbeddingsScorerTest(tf.test.TestCase):

  def test_all_options_work(self):
    actions = tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    contexts = tf.constant([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    x = (actions, contexts)
    dot = stateless.EmbeddingsScorer(SF.SF_DOT)(x)
    omdot = stateless.EmbeddingsScorer(SF.SF_OMDOT)(x)
    hardmin = stateless.EmbeddingsScorer(SF.SF_HARDMIN)(x)
    softmin = stateless.EmbeddingsScorer(SF.SF_SOFTMIN)(x)
    self.assertAllClose(dot, [3.0, 2.0])
    self.assertAllClose(omdot, [1.0 - 3.0, 1.0 - 2.0])
    self.assertAllClose(hardmin, [2.0, 1.0])
    self.assertAllClose(softmin, [1 / 2 + 2 / 3, 2 / 3])

  def test_handles_all_enum_values(self):
    actions = tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    contexts = tf.constant([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    x = (actions, contexts)
    for scoring_function in SF.values():
      if scoring_function == SF.SF_UNSPECIFIED:
        continue
      y = stateless.EmbeddingsScorer(scoring_function)(x)
      self.assertNotEmpty(y)

  def test_serializes(self):
    config = {'scoring_function': SF.SF_HARDMIN}
    f = stateless.EmbeddingsScorer.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(stateless.EmbeddingsScorer).parameters.keys(),
        config.keys(),
    )


class ContrastiveLabelsScoresTest(tf.test.TestCase):

  def test_serializes(self):
    config = {
        'scoring_function': SF.SF_DOT,
        'contrastive_scores_per_query': 42,
        'positive_instances_weight_factor': 1.0,
    }
    f = stateless.ContrastiveLabelsScores.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(stateless.ContrastiveLabelsScores).parameters.keys(),
        config.keys(),
    )

  def test_handles_empty_queries_items(self):
    f = stateless.ContrastiveLabelsScores(SF.SF_DOT, 1000, 1.0)
    inputs = {
        'queries': tf.constant([[]], dtype=tf.float32, shape=[0, 1]),
        'items': tf.constant([[0.0], [1.0]]),
        'matching_items': tf.constant([], dtype=tf.int32),
    }
    labels, scores, weights = f(inputs)
    self.assertEmpty(labels)
    self.assertEmpty(scores)
    self.assertEmpty(weights)

  def test_handles_int64_matching_items(self):
    f = stateless.ContrastiveLabelsScores(SF.SF_DOT, 1, 1.0)
    inputs = {
        'queries': tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]),
        'items': tf.constant([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
        'matching_items': tf.constant([0, 1], dtype=tf.int64),
    }
    labels, scores, weights = f(inputs)
    self.assertAllEqual(labels[:2], [-1, -1])
    self.assertAllEqual(scores[:2], [3.0, 2.0])
    self.assertAllEqual(weights[:2], [1.0, 1.0])

  def test_handles_zero_positive_scores(self):
    f = stateless.ContrastiveLabelsScores(SF.SF_DOT, 0, 1.0)
    inputs = {
        'queries': tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]),
        'items': tf.constant(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        'matching_items': tf.constant([0, 1], dtype=tf.int32),
    }
    labels, scores, weights = f(inputs)
    self.assertAllEqual(labels, [-1, -1])
    self.assertAllClose(scores, [3.0, 2.0])
    self.assertAllEqual(weights, [1.0, 1.0])

  def test_marks_valid_pairs(self):
    f = stateless.ContrastiveLabelsScores(SF.SF_DOT, 1000, 1.0)
    inputs = {
        'queries': tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]),
        'items': tf.constant(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        'matching_items': tf.constant([0, 1], dtype=tf.int32),
    }
    labels, scores, weights = f(inputs)
    neg_labels = labels[:2]
    neg_scores = scores[:2]
    neg_weights = weights[:2]
    self.assertAllEqual(neg_labels, [-1, -1])
    self.assertAllClose(neg_scores, [3.0, 2.0])
    self.assertAllEqual(neg_weights, [1.0, 1.0])

    pos_labels = labels[2:]
    pos_scores = scores[2:]
    pos_weights = weights[2:]
    self.assertNotEmpty(pos_labels)
    self.assertShapeEqual(pos_labels, pos_scores)
    self.assertShapeEqual(pos_labels, pos_weights)
    self.assertSameElements(pos_labels, [1.0])
    true_pos_scores = pos_scores[pos_weights > 0]
    self.assertSameElements(true_pos_scores, [0.0, 1.0, 7.0])

  def test_respects_positive_size(self):
    f = stateless.ContrastiveLabelsScores(SF.SF_DOT, 43, 1.0)
    inputs = {
        'queries': tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]),
        'items': tf.constant(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        'matching_items': tf.constant([0, 1], dtype=tf.int32),
        # Compatibility keys are such that no pair is spurious.
        'query_compatibility_keys': tf.constant(['a', 'b']),
        'item_compatibility_keys': tf.constant(['c', 'd', 'e']),
    }
    labels, scores, weights = f(inputs)
    self.assertLen(labels, 2 + 43 * 2)
    self.assertLen(scores, 2 + 43 * 2)
    self.assertAllEqual(weights, [1.0] * (2 + 43 * 2))

  def test_respects_compatibility_keys(self):
    f = stateless.ContrastiveLabelsScores(SF.SF_DOT, 1000, 1.0)
    inputs = {
        'queries': tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]),
        'items': tf.constant(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        'matching_items': tf.constant([0, 1], dtype=tf.int32),
        # Compatibility keys are such that the 0-2 pair is ruled out.
        'query_compatibility_keys': tf.constant(['a', 'b']),
        'item_compatibility_keys': tf.constant(['a', 'b', 'a']),
    }
    labels, scores, weights = f(inputs)
    neg_labels = labels[:2]
    neg_scores = scores[:2]
    neg_weights = weights[:2]
    self.assertAllEqual(neg_labels, [-1, -1])
    self.assertAllClose(neg_scores, [3.0, 2.0])
    self.assertAllEqual(neg_weights, [1.0, 1.0])

    pos_labels = labels[2:]
    pos_scores = scores[2:]
    pos_weights = weights[2:]
    self.assertNotEmpty(pos_labels)
    self.assertShapeEqual(pos_labels, pos_scores)
    self.assertShapeEqual(pos_labels, pos_weights)
    self.assertSameElements(pos_labels, [1.0])
    true_pos_scores = pos_scores[pos_weights > 0]
    self.assertSameElements(true_pos_scores, [1.0, 7.0])

  def test_explicit_weights(self):
    f = stateless.ContrastiveLabelsScores(SF.SF_DOT, 1000, 1.0)
    inputs = {
        'queries': tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]),
        'items': tf.constant(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        'matching_items': tf.constant([0, 1], dtype=tf.int32),
        'query_weights': tf.constant([1.0, 2.0]),
        'item_weights': tf.constant([3.0, 5.0, 7.0]),
    }
    labels, scores, weights = f(inputs)
    self.assertAllEqual(labels[:2], [-1, -1])
    self.assertAllEqual(scores[:2], [3.0, 2.0])
    self.assertAllEqual(weights[:2], [3.0, 10.0])
    possible_score_weights = [(1.0, 5.0), (0.0, 7.0), (7.0, 6.0), (1.0, 14.0)]
    seen_pairs = []
    for i, pair in enumerate(list(zip(scores[2:], weights[2:]))):
      if weights[2 + i] > 0.0:
        seen_pairs.append(pair)
    self.assertSameElements(seen_pairs, possible_score_weights)

  def test_rescales_positives(self):
    f = stateless.ContrastiveLabelsScores(SF.SF_DOT, 100, 0.5)
    inputs = {
        'queries': tf.constant([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]),
        'items': tf.constant(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        'matching_items': tf.constant([0, 1], dtype=tf.int32),
        # Compatibility keys are such that the 0-2 pair is ruled out.
        'query_compatibility_keys': tf.constant(['a', 'b']),
        'item_compatibility_keys': tf.constant(['a', 'b', 'a']),
    }
    _, _, weights = f(inputs)
    neg_weights = weights[:2]
    self.assertAllEqual(neg_weights, [1.0, 1.0])

    pos_weights = weights[2:]
    self.assertSameElements(pos_weights, [0.0, 0.5])


if __name__ == '__main__':
  tf.test.main()
