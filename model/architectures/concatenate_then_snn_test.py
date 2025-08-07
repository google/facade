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

from model.architectures import concatenate_then_snn as setsnn
from protos import config_pb2


# Functions used for testing readability.
def _features(*features):
  return list(features)


def _actions(*actions):
  return list(actions)


def _users(*users):
  return list(users)


class ConcatenateThenSNNTowerTest(tf.test.TestCase):

  def test_serializes(self):
    action_name = 'drive'
    segment_name = 'username'
    model_config = config_pb2.ModelHyperparameters(
        embedding_dims=100,
        action_embeddings_transformations=[
            config_pb2.Transformation.TR_SOFTPLUS
        ],
    )
    drive_config = model_config.action_name_to_architecture[
        action_name
    ].concatenate_then_snn
    drive_config.snn.layer_sizes.append(42)
    drive_seg_config = drive_config.segment_reductions.add()
    drive_seg_config.token_feature_name = segment_name
    drive_seg_config.segment_weight_scaling = config_pb2.SegmentReduction.WS_LOG
    drive_seg_config.segment_weight_normalization = (
        config_pb2.SegmentReduction.WN_L2
    )
    drive_seg_config.token_embedding_name = 'usernames'
    model_config.token_embedding_name_to_config['usernames'].dimensions = 50
    f = setsnn.ConcatenateThenSNNTower(model_config, action_name)
    setsnn.ConcatenateThenSNNTower.from_config(f.get_config())

  def test_context_tower_segment_reductions_only(self):
    embedding_dims = 128
    model_config = config_pb2.ModelHyperparameters(
        embedding_dims=embedding_dims,
        context_embeddings_transformations=[
            config_pb2.Transformation.TR_SOFTPLUS
        ],
    )
    embedding_id = 'usernames'
    model_config.token_embedding_name_to_config[embedding_id].dimensions = 3

    architecture = model_config.context_architecture.concatenate_then_snn
    architecture.snn.layer_sizes.append(42)
    feature1 = 'critique_peers'
    reduction = architecture.segment_reductions.add()
    reduction.token_feature_name = feature1
    reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_LOG
    reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
    reduction.token_embedding_name = embedding_id

    feature2 = 'calendar_peers'
    feature2i = 'calendar_peers_intensity'
    reduction = architecture.segment_reductions.add()
    reduction.token_feature_name = feature2
    reduction.intensity_feature_name = feature2i
    reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_IDENTITY
    reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
    reduction.token_embedding_name = embedding_id

    tower = setsnn.ConcatenateThenSNNTower(model_config)

    # Three contexts with features of varying sizes.
    # Feature 1 tokens: a, b, c. Feature 2 tokens: d, e, f.
    # Context 1: {segment 1: [a, b] segment 2: []}
    # Context 2: {segment 1: []     segment 2: [d]}
    # Context 3: {segment 1: [c]    segment 2: [e, f]}
    t_a = [0.0, 0.0, 1.0]
    t_b = [0.0, 1.0, 0.0]
    t_c = [1.0, 0.0, 0.0]
    t_d = [0.0, 1.0, 1.0]
    t_e = [1.0, 0.0, 1.0]
    t_f = [1.0, 1.0, 0.0]
    inputs = {
        feature1: tf.ragged.constant(
            _users(_features(t_a, t_b), _features(), _features(t_c)),
            ragged_rank=1,
        ),
        feature2: tf.ragged.constant(
            _users(_features(), _features(t_d), _features(t_e, t_f)),
            ragged_rank=1,
        ),
        feature2i: tf.ragged.constant(
            _users(_features(), _features(1.0), _features(2.0, 3.0))
        ),
    }

    context_embeddings = tower(inputs)
    self.assertAllEqual(context_embeddings.shape, [3, embedding_dims])
    # We specified a softplus final transformation.
    self.assertAllGreaterEqual(context_embeddings, 0.0)

  def test_action_tower_segment_reductions_only(self):
    embedding_dims = 128
    model_config = config_pb2.ModelHyperparameters(
        embedding_dims=embedding_dims,
        action_embeddings_transformations=[
            config_pb2.Transformation.TR_SOFTPLUS
        ],
    )
    embedding_id = 'usernames'
    model_config.token_embedding_name_to_config[embedding_id].dimensions = 3

    action_name = 'drive'
    architecture = model_config.action_name_to_architecture[
        action_name
    ].concatenate_then_snn
    architecture.snn.layer_sizes.append(42)
    feature1 = 'viewers'
    reduction = architecture.segment_reductions.add()
    reduction.token_feature_name = feature1
    reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_LOG
    reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
    reduction.token_embedding_name = embedding_id

    feature2 = 'writers'
    feature2i = 'writers_intensity'
    reduction = architecture.segment_reductions.add()
    reduction.token_feature_name = feature2
    reduction.intensity_feature_name = feature2i
    reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_IDENTITY
    reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
    reduction.token_embedding_name = embedding_id

    tower = setsnn.ConcatenateThenSNNTower(model_config, action_name)

    # Three actions with two features belonging to two contexts.
    # Feature 1 tokens: a, b, c. Feature 2 tokens: d, e, f.
    # Context 1: {action 1: {segment 1: [a c] segment 2: [e f]}}
    #            {action 2: {segment 1: [b] segment 2: [d]}}
    # Context 2: {action 3: {segment 1: [] segment 2: []}}
    t_a = [0.0, 0.0, 1.0]
    t_b = [0.0, 1.0, 0.0]
    t_c = [1.0, 0.0, 0.0]
    t_d = [0.0, 1.0, 1.0]
    t_e = [1.0, 0.0, 1.0]
    t_f = [1.0, 1.0, 0.0]
    inputs = {
        feature1: tf.ragged.constant(
            _users(
                _actions(_features(t_a, t_c), _features(t_b)),
                _actions(_features()),
            ),
            ragged_rank=2,
        ),
        feature2: tf.ragged.constant(
            _users(
                _actions(_features(t_e, t_f), _features(t_d)),
                _actions(_features()),
            ),
            ragged_rank=2,
        ),
        feature2i: tf.ragged.constant(
            _users(
                _actions(_features(1.0, 2.0), _features(3.0)),
                _actions(_features()),
            )
        ),
    }

    action_embeddings = tower(inputs).to_list()
    self.assertLen(action_embeddings, 2)
    self.assertLen(action_embeddings[0], 2)
    self.assertLen(action_embeddings[1], 1)
    for by_context in action_embeddings:
      for e in by_context:
        self.assertLen(e, embedding_dims)
        self.assertAllGreaterEqual(e, 0.0)

  def test_context_tower_dense_features_only(self):
    embedding_dims = 128
    model_config = config_pb2.ModelHyperparameters(
        embedding_dims=embedding_dims,
        context_embeddings_transformations=[
            config_pb2.Transformation.TR_SOFTPLUS
        ],
    )

    architecture = model_config.context_architecture.concatenate_then_snn
    architecture.snn.layer_sizes.append(42)

    feature1 = 'user_dense_1'
    dense = architecture.fixed_size_dense_features.add()
    dense.feature_name = feature1
    dense.size = 4

    feature2 = 'user_dense_2'
    dense = architecture.fixed_size_dense_features.add()
    dense.feature_name = feature2
    dense.size = 6

    tower = setsnn.ConcatenateThenSNNTower(model_config)

    # Two contexts with embedding vectors of varying sizes.
    # context 1: {segment 1: [4-dim]   segment 2: [6-dim]}
    # context 2: {segment 1: [4-dim]   segment 2: [6-dim]}
    inputs = {
        feature1: tf.ragged.constant(
            _users([0.2, 0.1, 0.3, 0.2], [0.0, 1.0, 0.3, 0.8]), ragged_rank=0
        ),
        feature2: tf.ragged.constant(
            _users(
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
            ),
            ragged_rank=0,
        ),
    }

    context_embeddings = tower(inputs)
    self.assertAllEqual(context_embeddings.shape, [2, embedding_dims])
    # We specified a softplus final transformation.
    self.assertAllGreaterEqual(context_embeddings, 0.0)

  def test_action_tower_dense_features_only(self):
    embedding_dims = 128
    model_config = config_pb2.ModelHyperparameters(
        embedding_dims=embedding_dims,
        action_embeddings_transformations=[
            config_pb2.Transformation.TR_SOFTPLUS
        ],
    )
    action_name = 'linhelm'
    architecture = model_config.action_name_to_architecture[
        action_name
    ].concatenate_then_snn
    architecture.snn.layer_sizes.append(42)

    feature1 = 'action_dense_1'
    dense = architecture.fixed_size_dense_features.add()
    dense.feature_name = feature1
    dense.size = 3

    feature2 = 'action_dense_2'
    dense = architecture.fixed_size_dense_features.add()
    dense.feature_name = feature2
    dense.size = 2

    tower = setsnn.ConcatenateThenSNNTower(model_config, action_name)

    # Three actions with two features belonging to two contexts.
    # context 1: {action 1: {segment 1: [3-dim] segment 2: [2-dim]}}
    #            {action 2: {segment 1: [3-dim] segment 2: [2-dim]}}
    # context 2: {action 3: {segment 1: [3-dim] segment 2: [2-dim]}}
    inputs = {
        feature1: tf.ragged.constant(
            _users(
                _actions([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
                _actions([1.0, 0.0, 0.0]),
            ),
            ragged_rank=1,
        ),
        feature2: tf.ragged.constant(
            _users(_actions([0.0, 1.0], [1.0, 0.0]), _actions([1.0, 1.0])),
            ragged_rank=1,
        ),
    }

    action_embeddings = tower(inputs).to_list()
    self.assertLen(action_embeddings, 2)
    self.assertLen(action_embeddings[0], 2)
    self.assertLen(action_embeddings[1], 1)
    for by_context in action_embeddings:
      for e in by_context:
        self.assertLen(e, embedding_dims)
        self.assertAllGreaterEqual(e, 0.0)

  def test_context_tower_mixed_segments(self):
    embedding_dims = 128
    model_config = config_pb2.ModelHyperparameters(
        embedding_dims=embedding_dims,
        context_embeddings_transformations=[
            config_pb2.Transformation.TR_SOFTPLUS
        ],
    )

    embedding_id = 'usernames'
    model_config.token_embedding_name_to_config[embedding_id].dimensions = 3

    architecture = model_config.context_architecture.concatenate_then_snn
    architecture.snn.layer_sizes.append(42)

    feature1 = 'user_dense_1'
    dense = architecture.fixed_size_dense_features.add()
    dense.feature_name = feature1
    dense.size = 4

    feature2 = 'calendar_peers'
    feature2i = 'calendar_peers_intensity'
    reduction = architecture.segment_reductions.add()
    reduction.token_feature_name = feature2
    reduction.intensity_feature_name = feature2i
    reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_IDENTITY
    reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
    reduction.token_embedding_name = embedding_id

    tower = setsnn.ConcatenateThenSNNTower(model_config)

    # Two contexts with embedding vectors of varying sizes.
    # context 1: {segment 1: [4-dim]   segment 2: [a, b]}
    # context 2: {segment 1: [4-dim]   segment 2: []}
    inputs = {
        feature1: tf.ragged.constant(
            _users([0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.3, 0.8]), ragged_rank=0
        ),
        feature2: tf.ragged.constant(
            _users(_features(), _features([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])),
            ragged_rank=1,
        ),
        feature2i: tf.ragged.constant(_users(_features(), _features(2.0, 3.0))),
    }

    context_embeddings = tower(inputs)
    self.assertAllEqual(context_embeddings.shape, [2, embedding_dims])
    # We specified a softplus final transformation.
    self.assertAllGreaterEqual(context_embeddings, 0.0)

  def test_action_tower_mixed_segments(self):
    embedding_dims = 128
    model_config = config_pb2.ModelHyperparameters(
        embedding_dims=embedding_dims,
        action_embeddings_transformations=[
            config_pb2.Transformation.TR_SOFTPLUS
        ],
    )
    embedding_id = 'usernames'
    model_config.token_embedding_name_to_config[embedding_id].dimensions = 3

    action_name = 'linhelm'
    architecture = model_config.action_name_to_architecture[
        action_name
    ].concatenate_then_snn
    architecture.snn.layer_sizes.append(42)

    feature1 = 'action_dense'
    dense = architecture.fixed_size_dense_features.add()
    dense.feature_name = feature1
    dense.size = 3

    feature2 = 'action_sparse'
    reduction = architecture.segment_reductions.add()
    reduction.token_feature_name = feature2
    reduction.segment_weight_scaling = config_pb2.SegmentReduction.WS_IDENTITY
    reduction.segment_weight_normalization = config_pb2.SegmentReduction.WN_L2
    reduction.token_embedding_name = embedding_id

    tower = setsnn.ConcatenateThenSNNTower(model_config, action_name)

    # Three actions with two features belonging to two contexts.
    # context 1: {action 1: {segment 1: [3-dim] segment 2: [e f]}}
    #            {action 2: {segment 1: [3-dim] segment 2: [d]}}
    # context 2: {action 3: {segment 1: [3-dim] segment 2: []}}
    t_d = [0.0, 1.0, 1.0]
    t_e = [1.0, 0.0, 1.0]
    t_f = [1.0, 1.0, 0.0]
    inputs = {
        feature1: tf.ragged.constant(
            _users(
                _actions([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
                _actions([1.0, 0.0, 0.0]),
            ),
            ragged_rank=1,
        ),
        feature2: tf.ragged.constant(
            _users(
                _actions(_features(t_e, t_f), _features(t_d)),
                _actions(_features()),
            ),
            ragged_rank=2,
        ),
    }

    action_embeddings = tower(inputs).to_list()
    self.assertLen(action_embeddings, 2)
    self.assertLen(action_embeddings[0], 2)
    self.assertLen(action_embeddings[1], 1)
    for by_context in action_embeddings:
      for e in by_context:
        self.assertLen(e, embedding_dims)
        self.assertAllGreaterEqual(e, 0.0)


if __name__ == '__main__':
  tf.test.main()
