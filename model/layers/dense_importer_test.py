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

from model.layers import dense_importer


# Functions used for testing readability.
def _actions(*actions):
  return list(actions)


def _users(*users):
  return list(users)


class DenseImporterTest(tf.test.TestCase):

  def test_context_particles_works(self):
    f = dense_importer.DenseImporter(5)
    # Make a batch of 2 contexts.
    u1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    u2 = [0.0, 0.0, 0.0, 0.0, 0.0]
    dense_vector = tf.constant(_users(u1, u2))

    y0 = f({'dense_vector': dense_vector})

    self.assertIsInstance(y0, tf.Tensor)
    self.assertEqual(y0.shape, (2, 5))
    self.assertAllClose(y0[0, :], np.zeros(5))
    self.assertAllClose(y0[1, :], np.zeros(5))

  def test_context_particles_works_in_graph_mode(self):
    f = dense_importer.DenseImporter(5)
    f_graph = tf.function(f)
    # Make a batch of 2 contexts.
    u1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    u2 = [0.0, 0.0, 0.0, 0.0, 0.0]
    dense_vector = tf.constant(_users(u1, u2))

    y0 = f_graph({'dense_vector': dense_vector})

    self.assertIsInstance(y0, tf.Tensor)
    self.assertEqual(y0.shape, (2, 5))
    self.assertAllClose(y0[0, :], np.zeros(5))
    self.assertAllClose(y0[1, :], np.zeros(5))

  def test_context_particles_errors_when_size_mismatch(self):
    f = dense_importer.DenseImporter(5)
    # Make a batch of 2 contexts of size 4.
    u1 = [0.0, 0.0, 0.0, 0.0]
    u2 = [0.0, 0.0, 0.0, 0.0]
    dense_vector = tf.constant(_users(u1, u2))
    with self.assertRaises(ValueError):
      f({'dense_vector': dense_vector})

  def test_context_particles_errors_when_size_mismatch_in_graph_mode(self):
    f = dense_importer.DenseImporter(5)
    f_graph = tf.function(f)
    u1 = [0.0, 0.0, 0.0, 0.0]
    u2 = [0.0, 0.0, 0.0, 0.0]
    dense_vector = tf.constant(_users(u1, u2))
    with self.assertRaises(ValueError):
      f_graph({'dense_vector': dense_vector})

  def test_action_particles_works(self):
    f = dense_importer.DenseImporter(2)
    # Extra dimension to match action import format.
    e = [1.0, 2.0]
    u0 = _actions([e])
    u1 = _actions([e], [e], [e], [e])
    dense_vector = tf.ragged.constant(_users(u0, u1), ragged_rank=2)

    y = f({
        'dense_vector': dense_vector,
    })

    self.assertIsInstance(y, tf.RaggedTensor)
    y = y.to_list()
    # expected structure is [[e], [e, e, e, e]]
    self.assertLen(y, 2)
    self.assertLen(y[0], 1)
    self.assertAllClose(y[0][0], e)
    self.assertLen(y[1], 4)
    for element in y[1]:
      self.assertAllClose(element, e)

  def test_action_particles_works_in_graph_mode(self):
    f = dense_importer.DenseImporter(2)
    f_graph = tf.function(f)
    e = [1.0, 2.0]
    # Extra dimension to match action import format.
    u0 = _actions([e])
    u1 = _actions([e], [e], [e], [e])
    dense_vector = tf.ragged.constant(_users(u0, u1), ragged_rank=2)

    y = f_graph({
        'dense_vector': dense_vector,
    })

    self.assertIsInstance(y, tf.RaggedTensor)
    y = y.to_list()
    # expected structure is [[e], [e, e, e, e]]
    self.assertLen(y, 2)
    self.assertLen(y[0], 1)
    self.assertAllClose(y[0][0], e)
    self.assertLen(y[1], 4)
    for element in y[1]:
      self.assertAllClose(element, e)

  def test_serializes(self):
    config = {'dense_vector_size': 32}
    f = dense_importer.DenseImporter.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(dense_importer.DenseImporter).parameters.keys(),
        config.keys(),
    )


if __name__ == '__main__':
  tf.test.main()
