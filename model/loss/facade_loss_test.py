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

from model.loss import facade_loss
from protos import config_pb2


class FacadeLossTest(tf.test.TestCase):

  def test_serializes(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    ph = model_config.training_hyperparameters.loss_function.pairwise_huber
    ph.hard_margin = 1.0
    ph.norm_push = 1.0
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "drive"
    ] = 1.0
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    f = facade_loss.FacadeModelLoss(model_config)
    facade_loss.FacadeModelLoss.from_config(f.get_config())

  def test_fails_on_missing_loss_weight(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    ph = model_config.training_hyperparameters.loss_function.pairwise_huber
    ph.hard_margin = 1.0
    ph.norm_push = 1.0
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    # Dummy architecture for `drive` action.
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    loss_layer = facade_loss.FacadeModelLoss(model_config)

    inputs = {}
    inputs[facade_loss.PRINCIPAL_IDENTITIES_KEY] = tf.constant([1, 2])
    inputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    inputs["drive"] = tf.ragged.constant(
        [[[1.0, 1.0], [0.0, 1.0]], []], inner_shape=(2,)
    )
    with self.assertRaisesRegex(
        ValueError, 'Missing loss weight for action "drive"'
    ):
      loss_layer(inputs)

  def test_fails_on_positive_strategy_unset(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    ph = model_config.training_hyperparameters.loss_function.pairwise_huber
    ph.hard_margin = 1.0
    ph.norm_push = 1.0
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    with self.assertRaisesRegex(
        ValueError, "Unknown synthetic positives strategy"
    ):
      facade_loss.FacadeModelLoss(model_config)

  def test_fails_on_zero_contrastive_scores_per_query(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    ph = model_config.training_hyperparameters.loss_function.pairwise_huber
    ph.hard_margin = 1.0
    ph.norm_push = 1.0
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 0
    strategy.positive_instances_weight_factor = 1.0
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    with self.assertRaisesRegex(
        ValueError, "Expected positive contrastive_scores_per_query"
    ):
      facade_loss.FacadeModelLoss(model_config)

  def test_huber_loss_contrastive_sampling(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    ph = model_config.training_hyperparameters.loss_function.pairwise_huber
    ph.hard_margin = 1.0
    ph.norm_push = 1.0
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "drive"
    ] = 1.0
    # Dummy architecture for `drive` action.
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    loss_layer = facade_loss.FacadeModelLoss(model_config)

    inputs = {}
    inputs[facade_loss.PRINCIPAL_IDENTITIES_KEY] = tf.constant([1, 2])
    inputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    inputs["drive"] = tf.ragged.constant(
        [[[1.0, 1.0], [0.0, 1.0]], []], inner_shape=(2,)
    )
    loss = loss_layer(inputs)
    self.assertGreaterEqual(loss, 0.0)

  def test_logistic_contrastive_sampling(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    gl = (
        model_config.training_hyperparameters.loss_function.generalized_logistic
    )
    gl.hard_margin = 0.5
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "drive"
    ] = 1.0
    # Dummy architecture for `drive` action.
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    loss_layer = facade_loss.FacadeModelLoss(model_config)

    inputs = {}
    inputs[facade_loss.PRINCIPAL_IDENTITIES_KEY] = tf.constant([1, 2])
    inputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    inputs["drive"] = tf.ragged.constant(
        [[[1.0, 1.0], [0.0, 1.0]], []], inner_shape=(2,)
    )
    loss = loss_layer(inputs)
    self.assertGreaterEqual(loss, 0.0)

  def test_sanll_contrastive_sampling(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    sl = model_config.training_hyperparameters.loss_function.sanll
    sl.margin = 0.5
    sl.negative_push = 0.1
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "drive"
    ] = 1.0
    # Dummy architecture for `drive` action.
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    loss_layer = facade_loss.FacadeModelLoss(model_config)

    inputs = {}
    inputs[facade_loss.PRINCIPAL_IDENTITIES_KEY] = tf.constant([1, 2])
    inputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = tf.constant(
        [[2.0, 0.0], [0.0, 3.0]]
    )
    inputs["drive"] = tf.ragged.constant(
        [[[0.9, 0.9], [0.01, 0.9]], []], inner_shape=(2,)
    )
    loss = loss_layer(inputs)
    self.assertGreaterEqual(loss, 0.0)

  def test_ms_contrastive_sampling(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_OMDOT,
    )
    ms = model_config.training_hyperparameters.loss_function.multi_similarity
    ms.a = 1.0
    ms.b = 1.0
    ms.loc = 0.5
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "drive"
    ] = 1.0
    # Dummy architecture for `drive` action.
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    loss_layer = facade_loss.FacadeModelLoss(model_config)

    inputs = {}
    inputs[facade_loss.PRINCIPAL_IDENTITIES_KEY] = tf.constant([1, 2])
    inputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = tf.constant(
        [[2.0, 0.0], [0.0, 3.0]]
    )
    inputs["drive"] = tf.ragged.constant(
        [[[0.9, 0.9], [0.01, 0.9]], []], inner_shape=(2,)
    )
    loss = loss_layer(inputs)
    self.assertGreaterEqual(loss, 0.0)

  def test_rescaling_works(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_OMDOT,
    )
    gl = (
        model_config.training_hyperparameters.loss_function.generalized_logistic
    )
    gl.hard_margin = 0.5
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "drive"
    ] = 1.0
    # Dummy architecture for `drive` action.
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    model_config.training_hyperparameters.loss_function.log_rescaler = True
    model_config.training_hyperparameters.loss_function.linear_rescaler.offset = (
        -1.0
    )
    model_config.training_hyperparameters.loss_function.linear_rescaler.scale = (
        0.1
    )
    loss_layer = facade_loss.FacadeModelLoss(model_config)

    inputs = {}
    inputs[facade_loss.PRINCIPAL_IDENTITIES_KEY] = tf.constant([1, 2])
    inputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    inputs["drive"] = tf.ragged.constant(
        [[[1.0, 1.0], [0.0, 1.0]], []], inner_shape=(2,)
    )
    loss = loss_layer(inputs)
    self.assertGreaterEqual(loss, 0.0)

  def test_pairwise_huber_commensurable_scores(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    ph = model_config.training_hyperparameters.loss_function.pairwise_huber
    ph.hard_margin = 1.0
    ph.norm_push = 1.0
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    model_config.training_hyperparameters.commensurable_scores_across_actions = (
        True
    )
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "drive"
    ] = 1.0
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "uberproxy"
    ] = 2.0
    # Dummy architectures for `drive` and `uberproxy` actions.
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    model_config.action_name_to_architecture[
        "uberproxy"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    loss_layer = facade_loss.FacadeModelLoss(model_config)

    inputs = {}
    inputs[facade_loss.PRINCIPAL_IDENTITIES_KEY] = tf.constant([1, 2])
    inputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    inputs["drive"] = tf.ragged.constant(
        [[[1.0, 1.0], [0.0, 1.0]], []], inner_shape=(2,)
    )
    inputs["uberproxy"] = tf.ragged.constant(
        [[[1.0, 1.0]], [[1.0, 0.0]]], inner_shape=(2,)
    )
    loss = loss_layer(inputs)
    self.assertGreaterEqual(loss, 0.0)

  def test_logistic_commensurable_scores(self):
    model_config = config_pb2.ModelHyperparameters(
        scoring_function=config_pb2.ScoringFunction.SF_DOT,
    )
    gl = (
        model_config.training_hyperparameters.loss_function.generalized_logistic
    )
    gl.hard_margin = 1.0
    strategy = (
        model_config.training_hyperparameters.synthetic_positives_strategy.random_sample_within_minibatch
    )
    strategy.contrastive_scores_per_query = 10
    strategy.positive_instances_weight_factor = 1.0
    model_config.training_hyperparameters.commensurable_scores_across_actions = (
        True
    )
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "drive"
    ] = 1.0
    model_config.training_hyperparameters.action_name_to_loss_weight[
        "uberproxy"
    ] = 2.0
    # Dummy architectures for `drive` and `uberproxy` actions.
    model_config.action_name_to_architecture[
        "drive"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    model_config.action_name_to_architecture[
        "uberproxy"
    ].concatenate_then_snn.snn.layer_sizes.append(1)
    loss_layer = facade_loss.FacadeModelLoss(model_config)

    inputs = {}
    inputs[facade_loss.PRINCIPAL_IDENTITIES_KEY] = tf.constant([1, 2])
    inputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    inputs["drive"] = tf.ragged.constant(
        [[[1.0, 1.0], [0.0, 1.0]], []], inner_shape=(2,)
    )
    inputs["uberproxy"] = tf.ragged.constant(
        [[[1.0, 1.0]], [[1.0, 0.0]]], inner_shape=(2,)
    )
    loss = loss_layer(inputs)
    self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
  tf.test.main()
