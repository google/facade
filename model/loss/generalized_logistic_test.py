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

from model.loss import generalized_logistic as gl
from parameterized import parameterized


def _hinge_ref(
    t: np.ndarray, soft_margin: float, hard_margin: float
) -> np.ndarray:
  del soft_margin  # ignored.
  return np.maximum(0.0, hard_margin - t)


def _logistic_ref(
    t: np.ndarray, soft_margin: float, hard_margin: float
) -> np.ndarray:
  return soft_margin * np.log1p(np.exp((hard_margin - t) / soft_margin))


def _negative_push_ref(x: np.ndarray, p: float):
  return (1 + x) ** (1 + p) - 1


class GeneralizedLogisticTest(tf.test.TestCase):

  @parameterized.expand([
      [0.0, 0.1, 0.0],
      [0.0, 0.2, 0.0],
      [0.1, 0.2, 0.0],
      [0.1, 0.3, 0.0],
      [0.0, 0.1, 0.2],
      [0.0, 0.2, 0.2],
      [0.1, 0.2, 0.2],
      [0.1, 0.3, 0.2],
  ])
  def testBehavior(self, soft_margin, hard_margin, negative_push):
    f = gl.GeneralizedLogisticLoss(
        soft_margin=soft_margin,
        hard_margin=hard_margin,
        negative_push=negative_push,
    )
    scores = np.linspace(-10, 10, num=1000)
    y_pos = np.ones_like(scores)
    pos_losses = f.call(y_pos, scores)
    neg_losses = f.call(-y_pos, scores)

    if soft_margin <= 0:
      ref_loss = _hinge_ref
    else:
      ref_loss = _logistic_ref

    ref_pos_losses = ref_loss(scores, soft_margin, hard_margin)
    ref_neg_losses_no_push = ref_loss(-scores, soft_margin, hard_margin)
    ref_neg_losses = _negative_push_ref(ref_neg_losses_no_push, negative_push)
    # Sanity-check negative push by verifying monotonicity.
    self.assertAllGreaterEqual(ref_neg_losses - ref_neg_losses_no_push, -1e-9)
    # Sanity-check ref implementation by looking at the asymptotic behavior.
    self.assertAllClose(ref_pos_losses[-1], 0.0)
    self.assertGreater(ref_pos_losses[0], 5.0)
    self.assertAllClose(ref_neg_losses[0], 0.0)
    self.assertGreater(ref_neg_losses[-1], 5.0)
    self.assertAllGreaterEqual(ref_pos_losses, 0.0)
    self.assertAllGreaterEqual(ref_neg_losses, 0.0)

    self.assertAllClose(ref_pos_losses, pos_losses)
    self.assertAllClose(ref_neg_losses, neg_losses)

  def testSerializes(self):
    config = {
        'soft_margin': 0.1,
        'hard_margin': 0.2,
        'negative_push': 0.1,
    }
    f = gl.GeneralizedLogisticLoss.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(gl.GeneralizedLogisticLoss).parameters.keys(),
        config.keys(),
    )


if __name__ == '__main__':
  tf.test.main()
