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
import tensorflow as tf
from model.loss import pairwise_huber as ph
from parameterized import parameterized


def reference_implementation(
    neg,
    pos,
    soft_margin,
    hard_margin,
    norm_push,
    lse_scale,
    w_neg=None,
    w_pos=None,
):
  assert lse_scale == 0 or norm_push == 1
  if w_neg is None:
    w_neg = [1.0] * len(neg)
  if w_pos is None:
    w_pos = [1.0] * len(pos)
  w_neg = [w / sum(w_neg) for w in w_neg]
  w_pos = [w / sum(w_pos) for w in w_pos]

  def f(t):
    if soft_margin == 0:
      if t < -hard_margin:
        return 0.0
      else:
        return t + hard_margin

    if t < -soft_margin / 2 - hard_margin:
      return 0.0
    if t < soft_margin / 2 - hard_margin:
      return 1 / (2 * soft_margin) * (t + soft_margin / 2 + hard_margin) ** 2
    return t + hard_margin

  loss = 0.0
  for wn, n in zip(w_neg, neg):
    instance_loss = 0.0
    for wp, p in zip(w_pos, pos):
      instance_loss += wp * f(n - p)
    if lse_scale == 0.0:
      instance_loss = instance_loss**norm_push
    else:
      instance_loss = math.exp(lse_scale * instance_loss)
    loss += wn * instance_loss

  if lse_scale == 0.0:
    return loss ** (1 / norm_push)
  else:
    return 1 / lse_scale * math.log(loss)


def cross_product(a, b):
  res = []
  for x in a:
    for y in b:
      res.append(x + y)
  return res


SCORES = [
    [[-1, 3, 0, -5, 2, 1], [10, 0, 4, -3, -1]],
    [
        [0.06218519, 1.41207531, 0.71445409, -0.25891416, 0.13114542],
        [1.66968299, 0.93206648, -0.87919924, 0.53612511, -1.08566769],
    ],
]
MARGINS_PUSH = [
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.1, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 2.0, 0.0],
    [0.1, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
]


class PairwiseHuberLossTest(tf.test.TestCase):

  @parameterized.expand(
      cross_product(
          [[[0], [0]], [[0], [1]], [[], []]], [[1, 0], [2, 0], [1, 1], [1, 2]]
      )
  )
  def testZeroOnEmptyInput(self, scores, labels, norm_push, lse_scale):
    y = tf.constant(scores, dtype=tf.float32)
    l = tf.constant(labels, dtype=tf.float32)
    soft_margin = 1.0
    hard_margin = 1.0
    loss = ph.PairwiseHuberLoss(soft_margin, hard_margin, norm_push, lse_scale)
    self.assertAlmostEqual(loss((l, y)).numpy(), 0.0, places=2)

  @parameterized.expand(cross_product(SCORES, MARGINS_PUSH))
  def testComputesLossCorrectly(
      self,
      negatives,
      positives,
      soft_margin,
      hard_margin,
      norm_push,
      lse_scale,
  ):
    y_negatives = tf.constant(negatives, dtype=tf.float32)
    y_positives = tf.constant(positives, dtype=tf.float32)
    y = tf.concat([y_negatives, y_positives], axis=0)
    l = tf.concat(
        [tf.zeros_like(y_negatives), tf.ones_like(y_positives)], axis=0
    )
    loss = ph.PairwiseHuberLoss(soft_margin, hard_margin, norm_push, lse_scale)
    self.assertAlmostEqual(
        loss((l, y)).numpy(),
        reference_implementation(
            negatives,
            positives,
            soft_margin,
            hard_margin,
            norm_push,
            lse_scale,
        ),
        places=5,
    )

  def testComputesWeightedLossCorrectly(self):
    negatives = SCORES[1][0]
    positives = SCORES[1][1]
    y_negatives = tf.constant(negatives, dtype=tf.float32)
    y_positives = tf.constant(positives, dtype=tf.float32)
    y = tf.concat([y_negatives, y_positives], axis=0)
    l = tf.concat(
        [tf.zeros_like(y_negatives), tf.ones_like(y_positives)], axis=0
    )
    w_neg = [0.2, 0.75, 0.3, 0.9, 0.3]
    w_pos = [0.28, 0.0, 0.08, 0.9, 0.18]  # Has one 0.0 weight.
    w = tf.concat([tf.constant(w_neg), tf.constant(w_pos)], axis=0)
    soft_margin, hard_margin, norm_push, lse_scale = MARGINS_PUSH[5]
    loss = ph.PairwiseHuberLoss(soft_margin, hard_margin, norm_push, lse_scale)
    self.assertAlmostEqual(
        loss((l, y, w)).numpy(),
        reference_implementation(
            negatives,
            positives,
            soft_margin,
            hard_margin,
            norm_push,
            lse_scale,
            w_neg=w_neg,
            w_pos=w_pos,
        ),
        places=5,
    )

  @parameterized.expand([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
  def testFiniteGradientAtZeroLoss(self, norm_push, lse_scale):
    soft_margin = 1.0
    hard_margin = 1.0
    loss = ph.PairwiseHuberLoss(soft_margin, hard_margin, norm_push, lse_scale)

    labels = tf.constant([0.0, 1.0], dtype=tf.float32)
    scores = tf.constant([-10.0, 10.0], dtype=tf.float32)
    with tf.GradientTape() as tape:
      tape.watch(scores)
      value = loss((labels, scores))
    grads = tape.gradient(value, scores)
    self.assertAllEqual(grads, [0.0, 0.0])

  def testDTypeMatters(self):
    loss_fn_64 = ph.PairwiseHuberLoss(0.0, 0.0, 1.0, 0.0, tf.float64)
    loss_fn_32 = ph.PairwiseHuberLoss(0.0, 0.0, 1.0, 0.0, tf.float32)
    labels = tf.concat(
        [
            tf.ones(10000000, dtype=tf.float32),
            -tf.ones(10000000, dtype=tf.float32),
        ],
        axis=0,
    )
    scores = tf.concat(
        [
            tf.zeros(10000000, dtype=tf.float32) * (1),
            tf.ones(10000000, dtype=tf.float32) * (1e6),
        ],
        axis=0,
    )
    weights = tf.ones(20000000, dtype=tf.float32)

    # The true value is 1e6.
    true_loss = 1e6
    loss_64 = loss_fn_64((labels, scores, weights))
    self.assertNear(loss_64 / true_loss, 1.0, 1e-6)

    # tf.float32 loses precision.
    loss_32 = loss_fn_32((labels, scores, weights))
    self.assertNotAllClose(loss_32 / true_loss, 1.0, 0.1)

  def testSerializes(self):
    config = {
        'soft_margin': 0.1,
        'hard_margin': 0.2,
        'norm_push': 1.0,
        'lse_scale': 0.4,
        'dtype': tf.float64,
    }
    f = ph.PairwiseHuberLoss.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(ph.PairwiseHuberLoss).parameters.keys(),
        config.keys(),
    )


if __name__ == '__main__':
  tf.test.main()
