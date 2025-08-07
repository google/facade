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
import os
import tempfile

from absl import flags
import tensorflow as tf

from model.architectures import token_embeddings
from model.common import test_utils
from protos import config_pb2


FEATURE_NAME_TO_VOCABULARY = {
    'drive/username': (False, [b'a', b'b']),
    'peers': (True, [b'b', b'c']),
}

VOCABULARY_FILE = ''


def setUpModule():
  global VOCABULARY_FILE
  VOCABULARY_FILE = test_utils.write_vocabulary_file(
      FEATURE_NAME_TO_VOCABULARY
  )


class TokenLookupEmbeddingsTest(tf.test.TestCase):

  def test_fails_on_missing_vocabulary(self):
    model_config = config_pb2.ModelHyperparameters()
    segment_reduction = (
        model_config.context_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction.token_feature_name = 'nonexistent_vocabulary'
    segment_reduction.token_embedding_name = 'id'
    model_config.token_embedding_name_to_config['id'].dimensions = 2

    with self.assertRaisesRegex(ValueError, 'nonexistent_vocabulary'):
      token_embeddings.TokenLookupEmbedders(VOCABULARY_FILE, model_config)

  def test_fails_on_missing_embedding_config(self):
    model_config = config_pb2.ModelHyperparameters()
    segment_reduction = (
        model_config.context_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction.token_feature_name = 'peers'
    segment_reduction.token_embedding_name = 'missing_embedding_id'

    with self.assertRaisesRegex(ValueError, 'missing_embedding_id'):
      token_embeddings.TokenLookupEmbedders(VOCABULARY_FILE, model_config)

  def test_makes_separate_embedders(self):
    model_config = config_pb2.ModelHyperparameters()
    segment_reduction1 = (
        model_config.context_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction1.token_feature_name = 'peers'
    segment_reduction1.token_embedding_name = 'id1'

    action_architecture = model_config.action_name_to_architecture['drive']
    segment_reduction2 = (
        action_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction2.token_feature_name = 'drive/username'
    segment_reduction2.token_embedding_name = 'id2'

    model_config.token_embedding_name_to_config['id1'].dimensions = 2
    embedding_config2 = model_config.token_embedding_name_to_config['id2']
    embedding_config2.dimensions = 3
    embedding_config2.num_oov_indices = 2

    tle = token_embeddings.TokenLookupEmbedders(VOCABULARY_FILE, model_config)
    context_features = {
        'drive/username': tf.constant(['id1', 'id2']),
        'peers': tf.constant(['id1', 'id2']),
    }
    sequence_features = {
        'drive/username': tf.constant(['id1', 'id2']),
        'peers': tf.constant(['id1', 'id2']),
    }

    c, s = tle.apply_string_lookup(context_features, sequence_features)
    context_features, sequence_features = tle.apply_embedders(c, s)

    self.assertNotAllClose(
        context_features['drive/username'], context_features['peers']
    )
    self.assertNotAllClose(
        sequence_features['drive/username'], sequence_features['peers']
    )

  def test_makes_shared_embedder(self):
    model_config = config_pb2.ModelHyperparameters()
    segment_reduction1 = (
        model_config.context_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction1.token_feature_name = 'peers'
    segment_reduction1.token_embedding_name = 'id'

    action_architecture = model_config.action_name_to_architecture['drive']
    segment_reduction2 = (
        action_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction2.token_feature_name = 'drive/username'
    segment_reduction2.token_embedding_name = 'id'

    model_config.token_embedding_name_to_config['id'].dimensions = 2

    tle = token_embeddings.TokenLookupEmbedders(VOCABULARY_FILE, model_config)
    context_features = {
        'drive/username': tf.constant(['id1', 'id2']),
        'peers': tf.constant(['id1', 'id2']),
    }
    sequence_features = {
        'drive/username': tf.constant(['id1', 'id2']),
        'peers': tf.constant(['id1', 'id2']),
    }

    c, s = tle.apply_string_lookup(context_features, sequence_features)
    context_features, sequence_features = tle.apply_embedders(c, s)

    self.assertAllClose(
        context_features['drive/username'], context_features['peers']
    )
    self.assertAllClose(
        sequence_features['drive/username'], sequence_features['peers']
    )

  def test_lookup(self):
    model_config = config_pb2.ModelHyperparameters()
    segment_reduction1 = (
        model_config.context_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction1.token_feature_name = 'peers'
    segment_reduction1.token_embedding_name = 'id'

    action_architecture = model_config.action_name_to_architecture['drive']
    segment_reduction2 = (
        action_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction2.token_feature_name = 'drive/username'
    segment_reduction2.token_embedding_name = 'id'

    model_config.token_embedding_name_to_config['id'].num_oov_indices = 1

    embedders = token_embeddings.TokenLookupEmbedders(
        VOCABULARY_FILE, model_config
    )

    feature_name1 = 'drive/username'
    feature_name2 = 'peers'
    context_inputs = {
        feature_name2: tf.sparse.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=tf.constant([b'c', b'd', b'b']),
            dense_shape=[13, 25],
        ),
        'untouched_feature': 42,
    }
    sequence_inputs = {
        feature_name1: tf.sparse.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=tf.constant([b'a', b'b', b'c']),
            dense_shape=[13, 25],
        ),
    }
    # Training with 0.0 oov_dropout_rate. No effect.
    context_result, sequence_result = embedders.apply_string_lookup(
        context_inputs, sequence_inputs, training=True
    )
    # Features without configs are passed through.
    self.assertEqual(context_result['untouched_feature'], 42)
    # Value of 0 indicates oov.
    self.assertAllEqual([1, 0, 2], context_result[feature_name2].values.numpy())
    # Order of values is deterministic based on hash.
    self.assertAllEqual(
        [3, 2, 1], sequence_result[feature_name1].values.numpy()
    )

    # Case with oov_dropout is almost 1.0.
    model_config.token_embedding_name_to_config['id'].oov_dropout_rate = 0.99999
    embedders_oov = token_embeddings.TokenLookupEmbedders(
        VOCABULARY_FILE, model_config
    )
    # Non training phase. No effect.
    context_result, sequence_result = embedders_oov.apply_string_lookup(
        context_inputs, sequence_inputs, training=False
    )
    # Features without configs are passed through.
    self.assertEqual(context_result['untouched_feature'], 42)
    # Value of 0 indicates oov.
    self.assertAllEqual([1, 0, 2], context_result[feature_name2].values.numpy())
    # Order of values is deterministic based on hash.
    self.assertAllEqual(
        [3, 2, 1], sequence_result[feature_name1].values.numpy()
    )

    # Training phase. All tokens are replaced with oov token 0.
    context_result, sequence_result = embedders_oov.apply_string_lookup(
        context_inputs, sequence_inputs, training=True
    )
    # Features without configs are passed through.
    self.assertEqual(context_result['untouched_feature'], 42)
    # All token ids are replaced with the oov id.
    self.assertAllEqual([0, 0, 0], context_result[feature_name2].values.numpy())
    self.assertAllEqual(
        [0, 0, 0], sequence_result[feature_name1].values.numpy()
    )

  def test_singular_embedder(self):
    model_config = config_pb2.ModelHyperparameters()
    segment_reduction = (
        model_config.context_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction.token_feature_name = 'peers'
    segment_reduction.token_embedding_name = 'id'

    model_config.token_embedding_name_to_config['id'].num_oov_indices = 1
    model_config.token_embedding_name_to_config['id'].dimensions = 10

    embedders = token_embeddings.TokenLookupEmbedders(
        VOCABULARY_FILE, model_config
    )

    feature_name = 'peers'
    inputs = {
        feature_name: tf.constant([0, 1, 2]),
        'untouched_feature': 42,
    }
    y, _ = embedders.apply_embedders(inputs, {})
    self.assertAllEqual(y['untouched_feature'], 42)
    self.assertAllEqual(y[feature_name].shape, [3, 10])

  def test_shared_embedder(self):
    model_config = config_pb2.ModelHyperparameters()
    segment_reduction1 = (
        model_config.context_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction1.token_feature_name = 'peers'
    segment_reduction1.token_embedding_name = 'id'

    action_architecture = model_config.action_name_to_architecture['drive']
    segment_reduction2 = (
        action_architecture.concatenate_then_snn.segment_reductions.add()
    )
    segment_reduction2.token_feature_name = 'drive/username'
    segment_reduction2.token_embedding_name = 'id'

    model_config.token_embedding_name_to_config['id'].dimensions = 10

    embedders = token_embeddings.TokenLookupEmbedders(
        VOCABULARY_FILE, model_config
    )

    feature_name1 = 'drive/username'
    feature_name2 = 'peers'
    context_inputs = {
        feature_name1: tf.constant([0]),
        'untouched_feature': 42,
    }
    sequence_inputs = {
        feature_name2: tf.constant([0, 1]),
    }
    context_result, sequence_result = embedders.apply_embedders(
        context_inputs, sequence_inputs
    )
    self.assertAllEqual(context_result['untouched_feature'], 42)
    self.assertAllEqual(context_result[feature_name1].shape, [1, 10])
    self.assertAllEqual(sequence_result[feature_name2].shape, [2, 10])
    self.assertAllClose(
        context_result[feature_name1], sequence_result[feature_name2][0:1, :]
    )


if __name__ == '__main__':
  tf.test.main()
