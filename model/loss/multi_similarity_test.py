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
import math

import numpy as np
import tensorflow as tf

from model.loss import multi_similarity


def _ref_impl(labels, scores, weights, a, b, loc):
  pos_acc = 0.0
  neg_acc = 0.0
  for score, label, weight in zip(scores, labels, weights):
    if label > 0:
      pos_acc += weight * math.exp(-a * (score - loc))
    else:
      neg_acc += weight * math.exp(b * (score - loc))
  return 1 / a * math.log(1 + pos_acc) + 1 / b * math.log(1 + neg_acc)


class MultiSimilarityTest(tf.test.TestCase):

  def test_correct_polarity(self):
    f = multi_similarity.MultiSimilarityLoss(a=1.0, b=1.0, loc=0.0)
    scores = np.array([-1000.0, 1000.0])
    labels = np.array([-1.0, 1.0])
    loss = f([labels, scores])
    self.assertAlmostEqual(loss, 0.0)

  def test_correct_asymptotic_behavior(self):
    f = multi_similarity.MultiSimilarityLoss(a=1.0, b=1.0, loc=0.0)
    scores = np.array([-1000.0, 2000.0])
    labels = np.array([1.0, -1.0])
    loss = f([labels, scores])
    self.assertAlmostEqual(loss, 1000.0 + 2000.0)

  def test_correct_value(self):
    a, b, loc = 1.0, 2.0, 0.5
    f = multi_similarity.MultiSimilarityLoss(a=a, b=b, loc=loc)
    scores = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
    labels = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    weights = np.array([1.0, 2.0, 0.7, 0.5, 1.1, 1.5, 1.0])
    loss = f([labels, scores, weights])
    self.assertAllClose(loss, _ref_impl(labels, scores, weights, a, b, loc))

  def test_handles_zero_weights(self):
    a, b, loc = 1.0, 2.0, 0.5
    f = multi_similarity.MultiSimilarityLoss(a=a, b=b, loc=loc)
    scores = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
    labels = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    weights = np.array([1.0, 0.0, 0.0, 0.5, 1.1, 1.5, 1.0])
    loss = f([labels, scores, weights])
    self.assertAllClose(loss, _ref_impl(labels, scores, weights, a, b, loc))

  def testSerializes(self):
    config = {
        'a': 1.0,
        'b': 2.0,
        'loc': 3.0,
    }
    f = multi_similarity.MultiSimilarityLoss.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(
            multi_similarity.MultiSimilarityLoss
        ).parameters.keys(),
        config.keys(),
    )


if __name__ == '__main__':
  tf.test.main()
