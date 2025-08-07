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

from model.optimization import sparse_sgd
from parameterized import parameterized


class SparseAndDenseModel(tf.keras.Model):
  """Deepset type model, mixing sparse and dense architectures."""

  def __init__(self, vocab_size: int, hidden_dims: int):
    super().__init__()
    self.embed = tf.keras.layers.Embedding(
        vocab_size,
        hidden_dims,
    )
    self.dense = tf.keras.layers.Dense(1)

  def call(self, inputs):
    embeddings = tf.math.reduce_sum(self.embed(inputs), axis=1)
    embeddings = tf.keras.activations.softplus(embeddings)
    return self.dense(embeddings)


def make_dataset(vocab_size: int, batch_size: int) -> tf.data.Dataset:
  """Whether a set of two uniformly random integers contains an even number."""

  def gen():
    while True:
      x = np.random.randint(0, vocab_size, size=(batch_size, 2), dtype=np.int32)
      y = np.where((np.prod(x, axis=1) % 2) == 0, 1.0, -1.0)
      yield x, y

  return tf.data.Dataset.from_generator(
      gen,
      output_signature=(
          tf.TensorSpec(shape=(batch_size, 2), dtype=tf.int32),
          tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
      ),
  ).repeat()


class SGDSpTest(tf.test.TestCase):

  def test_serializes(self):
    config = sparse_sgd.SGDSp(
        learning_rate=1.0, sparse_momentum=0.5, sparse_weight_decay=0.25
    ).get_config()
    print(config)
    f = sparse_sgd.SGDSp.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertContainsSubset(
        {'learning_rate', 'sparse_momentum', 'sparse_weight_decay'},
        inspect.signature(sparse_sgd.SGDSp).parameters.keys(),
    )

  def test_weight_decays(self):
    model = SparseAndDenseModel(10, 20)
    # Use a tiny learning rate to test all parameters are driven to zero.
    model.compile(
        optimizer=sparse_sgd.SGDSp(
            learning_rate=1e-6, sparse_weight_decay=1e6, sparse_momentum=0.0
        ),
        loss=tf.keras.losses.Hinge(),
    )

    data = make_dataset(10, 16)
    model.fit(data, epochs=1, steps_per_epoch=1000)
    for w in model.get_weights():
      self.assertAllClose(w, np.zeros_like(w))

  @parameterized.expand(
      [(0.1, 0.0, 0.0), (0.1, 1e-5, 0.0), (0.1, 0.0, 0.5), (0.1, 1e-4, 0.5)]
  )
  def test_optimizes(
      self, learning_rate: float, weight_decay: float, momentum: float
  ):
    model = SparseAndDenseModel(100, 10)
    model.compile(
        optimizer=sparse_sgd.SGDSp(learning_rate, weight_decay, momentum),
        loss=tf.keras.losses.Hinge(),
    )

    data = make_dataset(100, 16)
    initial_loss = model.evaluate(data, steps=1000)
    self.assertGreater(initial_loss, 0.5)

    model.fit(data, epochs=1, steps_per_epoch=1000)

    final_loss = model.evaluate(data, steps=1000)
    self.assertLessEqual(final_loss, 1e-3)


if __name__ == '__main__':
  tf.test.main()
