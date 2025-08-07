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

from parameterized import parameterized
from model.loss import sanll


def ref_neg_loss(scores, margin, negative_push):
  y = scores / margin
  return margin * (np.power(1 + y, 1 + negative_push) - 1)


def ref_pos_loss(scores, margin):
  y = scores / margin
  return -margin * np.log(sanll._EPSILON + 1 - np.exp(-y))


class SanllTest(tf.test.TestCase):

  @parameterized.expand([
      [0.1, 0.0],
      [0.1, 1.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [10.0, 0.0],
      [10.0, 1.0],
  ])
  def testBehavior(self, margin, negative_push):
    f = sanll.SanllLoss(
        margin=margin,
        negative_push=negative_push,
    )
    scores = np.linspace(0.0, 10.0, num=1000)
    y_pos = np.ones_like(scores)
    pos_losses = f.call(y_pos, scores)
    neg_losses = f.call(-y_pos, scores)

    ref_pos_losses = ref_pos_loss(scores, margin)
    ref_neg_losses_no_push = ref_neg_loss(scores, margin, 0.0)
    ref_neg_losses = ref_neg_loss(scores, margin, negative_push)
    # Sanity-check negative push by verifying monotonicity.
    self.assertAllGreaterEqual(ref_neg_losses - ref_neg_losses_no_push, -1e-9)
    # Sanity-check losses.
    self.assertAllGreaterEqual(ref_pos_losses, -1e-9)
    self.assertAllGreaterEqual(ref_neg_losses, 0.0)

    self.assertAllClose(ref_pos_losses, pos_losses)
    self.assertAllClose(ref_neg_losses, neg_losses)

  def testSerializes(self):
    config = {
        'margin': 0.1,
        'negative_push': 0.2,
    }
    f = sanll.SanllLoss.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(sanll.SanllLoss).parameters.keys(),
        config.keys(),
    )


if __name__ == '__main__':
  tf.test.main()
