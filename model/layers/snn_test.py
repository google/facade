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

import tensorflow as tf

from model.layers import snn


class SNNTest(tf.test.TestCase):

  def test_shapes(self):
    f = snn.SNN([10, 5, 10, 13], dropout_rate=0.5)
    x = tf.ones((2, 3), dtype=tf.float32)
    y = f(x, training=False)
    y_do = f(x, training=True)
    self.assertAllEqual(y.shape, y_do.shape)
    self.assertAllEqual(y.shape, (2, 13))

  def test_serializes(self):
    config = {'layer_sizes': [10, 5, 10, 13], 'dropout_rate': 0.5}
    f = snn.SNN.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(snn.SNN).parameters.keys(),
        config.keys(),
    )


if __name__ == '__main__':
  tf.test.main()
