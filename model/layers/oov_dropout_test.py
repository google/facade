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
import numpy as np
import tensorflow as tf
from parameterized import parameterized
from model.layers import oov_dropout


class OovDropoutTest(tf.test.TestCase):

  @parameterized.expand([
      (
          "normal_tensor_not_training",
          tf.constant(["foo", "bar", "baz"]),
          tf.constant([2, 3, 4]),
          False,
          0.999999999,
      ),
      (
          "ragged_tensor_not_training",
          tf.ragged.constant([["foo", "bar"], ["baz"]]),
          tf.ragged.constant([[2, 3], [4]]),
          False,
          0.999999999,
      ),
      (
          "normal_tensor_low_dropout_prob",
          tf.constant(["foo", "bar", "baz"]),
          tf.constant([2, 3, 4]),
          True,
          1e-12,
      ),
      (
          "ragged_tensor_low_dropout_prob",
          tf.ragged.constant([["foo", "bar"], ["baz"]]),
          tf.ragged.constant([[2, 3], [4]]),
          True,
          1e-12,
      ),
  ])
  def test_not_dropping(self, name, strings, expected_output, training, dropout_prob):
    model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=["foo", "bar", "baz"],
            num_oov_indices=2,
        ),
        oov_dropout.OovDropout(dropout_prob=dropout_prob, num_oov_tokens=2),
    ])

    output = model(strings, training=training)
    expected_output = tf.cast(expected_output, output.dtype)
    # Use reduce_all so Tensorflow can handle equality for ragged tensors.
    self.assertTrue(tf.reduce_all(output == expected_output).numpy())

  @parameterized.expand([
      ("normal_tensor", tf.constant(["foo", "bar", "baz"])),
      ("ragged_tensor", tf.ragged.constant([["foo", "bar"], ["baz"]])),
  ])
  def test_drops_tokens(self, name, strings):
    model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=["foo", "bar", "baz"],
            num_oov_indices=2,
        ),
        oov_dropout.OovDropout(dropout_prob=0.9999999999, num_oov_tokens=2),
    ])
    x = model(strings, training=True)

    if isinstance(x, tf.RaggedTensor):
      x = x.flat_values

    np.testing.assert_array_compare(np.less_equal, x.numpy(), 1)

  def test_sparse_input(self):
    model = oov_dropout.OovDropout(dropout_prob=0.9999999999, num_oov_tokens=2)
    sparse_input = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [0, 2]], values=[2, 3, 4], dense_shape=[2, 3]
    )
    x = model(sparse_input, training=True)
    x = tf.sparse.to_dense(x)

    np.testing.assert_array_compare(np.less_equal, x.numpy(), 1)


if __name__ == "__main__":
  tf.test.main()
