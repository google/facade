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
import tensorflow as tf
from model.loss import rescale
from protos import config_pb2


class RescaleTest(tf.test.TestCase):

  def test_serializes(self):
    config = config_pb2.LossFunction.LinearRescalerConfig()
    config.offset = -0.5
    config.scale = 1.0
    config.trainable_offset = True
    f = rescale.LinearRescaler(config)
    rescale.LinearRescaler.from_config(f.get_config())

  def test_fails_on_non_positive_scale(self):
    config = config_pb2.LossFunction.LinearRescalerConfig()
    with self.assertRaisesRegex(ValueError, "Scale must be positive"):
      rescale.LinearRescaler(config)

  def test_rescales(self):
    config = config_pb2.LossFunction.LinearRescalerConfig()
    config.offset = -0.5
    config.scale = 2.0
    f = rescale.LinearRescaler(config)
    x = tf.constant([-1.0, 0.0, 1.0])
    y = f(x)
    self.assertAllClose([-2.5, -0.5, 1.5], y)


if __name__ == "__main__":
  tf.test.main()
