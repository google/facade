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

from model.layers import segment_embedder
from protos import config_pb2


# Functions used for testing readability.
def _features(*features):
  return list(features)


def _actions(*actions):
  return list(actions)


def _users(*users):
  return list(users)


class SegmentEmbedderTest(tf.test.TestCase):

  def test_checks_for_negative_intensities(self):
    f = segment_embedder.SegmentEmbedder(
        weight_scaling=config_pb2.SegmentReduction.WeightScaling.WS_IDENTITY,
        weight_normalization=config_pb2.SegmentReduction.WeightNormalization.WN_L2,
        dropout_rate=0.5,
    )
    embeddings = tf.ragged.constant(
        _users(_actions(_features([1.0, 2.0]))), ragged_rank=2
    )
    intensities = tf.ragged.constant(_users(_actions(_features(-1.0))))

    with self.assertRaisesOpError(r'x >= 0'):
      f({
          'embeddings': embeddings,
          'intensities': intensities,
      })

  def test_action_l1_weight_norm(self):
    f = segment_embedder.SegmentEmbedder(
        weight_scaling=config_pb2.SegmentReduction.WeightScaling.WS_IDENTITY,
        weight_normalization=config_pb2.SegmentReduction.WeightNormalization.WN_L1,
        dropout_rate=0.0,
    )

    e = [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]
    u1 = _actions(_features(e[0]))
    u1i = _actions(_features(1.0))
    u2 = _actions(_features(e[0]), _features(e[1]))
    u2i = _actions(_features(1.0), _features(2.0))
    u3 = _actions(_features(e[0], e[1]), _features(e[2]))
    u3i = _actions(_features(1.0, 2.0), _features(3))
    embeddings = tf.ragged.constant(_users(u1, u2, u3), ragged_rank=2)
    intensities = tf.ragged.constant(_users(u1i, u2i, u3i))

    res_implicit_intens = f({
        'embeddings': embeddings,
    })
    res_with_intens = f({
        'embeddings': embeddings,
        'intensities': intensities,
    })

    res_implicit_intens = res_implicit_intens.to_list()
    res_with_intens = res_with_intens.to_list()
    e = np.array(e)
    expected_implicit_intens = [
        [e[0] / 2],
        [e[0] / 2, e[1] / 2],
        [(e[0] + e[1]) / 3, e[2] / 2],
    ]
    expected_with_intens = [
        [e[0] / 2],
        [e[0] / 2, e[1] * 2 / 3],
        [(e[0] + 2 * e[1]) / 4, e[2] * 3 / 4],
    ]
    self.assertAllClose(res_implicit_intens, expected_implicit_intens)
    self.assertAllClose(res_with_intens, expected_with_intens)

  def test_context_l1_weight_norm(self):
    f = segment_embedder.SegmentEmbedder(
        weight_scaling=config_pb2.SegmentReduction.WeightScaling.WS_IDENTITY,
        weight_normalization=config_pb2.SegmentReduction.WeightNormalization.WN_L1,
        dropout_rate=0.0,
    )

    e = [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]
    u1 = _features(e[0])
    u1i = _features(1.0)
    u2 = _features(e[0], e[1])
    u2i = _features(1.0, 2.0)
    embeddings = tf.ragged.constant(_users(u1, u2), ragged_rank=1)
    intensities = tf.ragged.constant(_users(u1i, u2i))

    res_implicit_intens = f({
        'embeddings': embeddings,
    })
    res_with_intens = f({
        'embeddings': embeddings,
        'intensities': intensities,
    })

    # Use .numpy() instead of .to_list() since these are now Tensors, as we've
    # reduced the only ragged dimension in the contexts.
    res_implicit_intens = res_implicit_intens.numpy()
    res_with_intens = res_with_intens.numpy()
    e = np.array(e)
    expected_implicit_intens = [
        e[0] / 2,
        (e[0] + e[1]) / 3,
    ]
    expected_with_intens = [e[0] / 2, (e[0] + 2 * e[1]) / 4]
    self.assertAllClose(res_implicit_intens, expected_implicit_intens)
    self.assertAllClose(res_with_intens, expected_with_intens)

  def test_action_log_weight_scaling(self):
    f = segment_embedder.SegmentEmbedder(
        weight_scaling=config_pb2.SegmentReduction.WeightScaling.WS_LOG,
        weight_normalization=config_pb2.SegmentReduction.WeightNormalization.WN_L1,
        dropout_rate=0.0,
    )

    w1 = math.log1p(1.0)
    w2 = math.log1p(2.0)
    w3 = math.log1p(3.0)

    e = [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]
    u1 = _actions(_features(e[0]))
    u1i = _actions(_features(1.0))
    u2 = _actions(_features(e[0]), _features(e[1]))
    u2i = _actions(_features(1.0), _features(2.0))
    u3 = _actions(_features(e[0], e[1]), _features(e[2]))
    u3i = _actions(_features(1.0, 2.0), _features(3))
    embeddings = tf.ragged.constant(_users(u1, u2, u3), ragged_rank=2)
    intensities = tf.ragged.constant(_users(u1i, u2i, u3i))

    res_implicit_intens = f({
        'embeddings': embeddings,
    })
    res_with_intens = f({
        'embeddings': embeddings,
        'intensities': intensities,
    })

    res_implicit_intens = res_implicit_intens.to_list()
    res_with_intens = res_with_intens.to_list()
    e = np.array(e)

    expected_implicit_intens = [
        [e[0] / 2],
        [e[0] / 2, e[1] / 2],
        [(e[0] + e[1]) / 3, e[2] / 2],
    ]
    expected_with_intens = [
        [e[0] / 2],
        [e[0] / 2, e[1] * w2 / (w1 + w2)],
        [(w1 * e[0] + w2 * e[1]) / (2 * w1 + w2), w3 * e[2] / (w1 + w3)],
    ]
    self.assertAllClose(res_implicit_intens, expected_implicit_intens)
    self.assertAllClose(res_with_intens, expected_with_intens)

  def test_particles_values_uniform_weight_scaling(self):
    f = segment_embedder.SegmentEmbedder(
        weight_scaling=config_pb2.SegmentReduction.WeightScaling.WS_UNIFORM,
        weight_normalization=config_pb2.SegmentReduction.WeightNormalization.WN_L1,
        dropout_rate=0.0,
    )

    e = [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]
    u1 = _actions(_features(e[0]))
    u1i = _actions(_features(1.0))
    u2 = _actions(_features(e[0]), _features(e[1]))
    u2i = _actions(_features(1.0), _features(2.0))
    u3 = _actions(_features(e[0], e[1]), _features(e[2]))
    u3i = _actions(_features(1.0, 2.0), _features(3))
    embeddings = tf.ragged.constant(_users(u1, u2, u3), ragged_rank=2)
    intensities = tf.ragged.constant(_users(u1i, u2i, u3i))

    res_implicit_intens = f({
        'embeddings': embeddings,
    })
    res_with_intens = f({
        'embeddings': embeddings,
        'intensities': intensities,
    })

    res_implicit_intens = res_implicit_intens.to_list()
    res_with_intens = res_with_intens.to_list()
    e = np.array(e)
    expected_implicit_intens = [
        [e[0] / 2],
        [e[0] / 2, e[1] / 2],
        [(e[0] + e[1]) / 3, e[2] / 2],
    ]
    self.assertAllClose(res_implicit_intens, expected_implicit_intens)
    self.assertAllClose(res_with_intens, expected_implicit_intens)

  def test_actions_with_total_dropout(self):
    f = segment_embedder.SegmentEmbedder(
        weight_scaling=config_pb2.SegmentReduction.WeightScaling.WS_UNIFORM,
        weight_normalization=config_pb2.SegmentReduction.WeightNormalization.WN_L1,
        dropout_rate=1,
    )
    e = [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]
    u1 = _actions(_features(e[0]))
    u1i = _actions(_features(1.0))
    u2 = _actions(_features(e[0]), _features(e[1]))
    u2i = _actions(_features(1.0), _features(2.0))
    u3 = _actions(_features(e[0], e[1]), _features(e[2]))
    u3i = _actions(_features(1.0, 2.0), _features(3))
    embeddings = tf.ragged.constant(_users(u1, u2, u3), ragged_rank=2)
    intensities = tf.ragged.constant(_users(u1i, u2i, u3i))

    res_implicit_intens = f(
        {
            'embeddings': embeddings,
        },
        training=True,
    )
    res_with_intens = f(
        {
            'embeddings': embeddings,
            'intensities': intensities,
        },
        training=True,
    )

    res_implicit_intens = res_implicit_intens.to_list()
    res_with_intens = res_with_intens.to_list()
    expected_implicit_intens = [
        [[0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    expected_with_intens = [
        [[0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    self.assertAllClose(res_implicit_intens, expected_implicit_intens)
    self.assertAllClose(res_with_intens, expected_with_intens)

  def test_serializes(self):
    config = {
        'weight_scaling': config_pb2.SegmentReduction.WeightScaling.WS_UNIFORM,
        'weight_normalization': (
            config_pb2.SegmentReduction.WeightNormalization.WN_L1
        ),
        'dropout_rate': 0.5,
    }
    f = segment_embedder.SegmentEmbedder.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(segment_embedder.SegmentEmbedder).parameters.keys(),
        config.keys(),
    )


if __name__ == '__main__':
  tf.test.main()
