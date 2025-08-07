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
"""Facade model definition: architecture and custom training step."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

from absl import logging
import tensorflow as tf
from tensorflow import keras

from google.protobuf import json_format
from model.architectures import concatenate_then_snn as setsnn
from model.architectures import token_embeddings as te
from model.input import pipeline
from model.layers import stateless
from model.loss import facade_loss
from model.metrics import combined
from protos import config_pb2


SCORES_KEY_SUFFIX = '_scores'
ALL_ACTIONS_METRIC_NAME = 'all_actions'

AnyTensor = tf.Tensor | tf.SparseTensor | tf.RaggedTensor
AnyMapping = Mapping[str, AnyTensor]


class FacadeModel(keras.Model):
  """Defines the Facade model.

  This class defines custom architecture, training, eval and inference logic.
  """

  def __init__(
      self,
      model_config: config_pb2.ModelHyperparameters,
      vocabulary_filepattern: str,
  ):
    """Instantiates a Facade model."""
    super().__init__()
    self.model_config = model_config
    self.vocabulary_filepattern = vocabulary_filepattern
    self.token_lookup_embedders = te.TokenLookupEmbedders(
        vocabulary_filepattern, model_config
    )
    self.context_tower = setsnn.ConcatenateThenSNNTower(model_config)
    self.action_towers = {}
    for action_name in sorted(model_config.action_name_to_architecture.keys()):
      tower = setsnn.ConcatenateThenSNNTower(model_config, action_name)
      self.action_towers[action_name] = tower
    self.scorer = stateless.EmbeddingsScorer(model_config.scoring_function)
    self.loss_layer = facade_loss.FacadeModelLoss(model_config)
    self.evaluation_metrics = None
    self._cached_metrics = None

  def get_config(self) -> Mapping[str, Any]:
    return {
        'model_config': json_format.MessageToJson(self.model_config),
        'vocabulary_filepattern': self.vocabulary_filepattern,
    }

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> 'FacadeModel':
    model_config = config_pb2.ModelHyperparameters()
    json_format.Parse(config['model_config'], model_config)
    vocabulary_filepattern = config['vocabulary_filepattern']
    return cls(model_config, vocabulary_filepattern)

  def call(
      self,
      inputs: tuple[AnyMapping, AnyMapping],
      training: Optional[bool] = None,
      mask: Optional[tf.Tensor] = None,
  ) -> Mapping[str, Union[tf.Tensor, tf.RaggedTensor]]:
    """Builds the inference graph.

    This returns a dictionary of named tensors, including:
      * context embeddings (as Tensor),
      * per-action type action embeddings (as RaggedTensors),
      * per-action type action scores (as Tensors).

    Args:
      inputs: The parsed input data as provided by
        http://google3/security/brainauth/facade2/model/input/pipeline.py
      training: Whether to build a training-time inference graph.
      mask: Unused, but required argument for Keras model serialization.

    Returns:
      outputs: A dictionary of named tensors representing the various outputs
        of the model.
    """
    del mask
    context_inputs, sequence_inputs = inputs
    context_batch, sequence_batch = self.token_lookup_embedders.apply_embedders(
        context_inputs, sequence_inputs
    )
    context_embeddings: tf.Tensor = self.context_tower(
        context_batch, training=training
    )
    outputs = {}
    outputs[facade_loss.CONTEXT_EMBEDDINGS_KEY] = context_embeddings
    for action_name, tower in sorted(self.action_towers.items()):
      action_embeddings: tf.RaggedTensor = tower(
          sequence_batch, training=training
      )
      outputs[action_name] = action_embeddings
      aligned_contexts = tf.gather(
          context_embeddings, action_embeddings.value_rowids()
      )
      scores = self.scorer((aligned_contexts, action_embeddings.flat_values))
      scores = action_embeddings.with_flat_values(scores)
      outputs[action_name + SCORES_KEY_SUFFIX] = scores

    return outputs

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
  def predict_from_serialized_examples(
      self, inputs: tf.Tensor
  ) -> Mapping[str, tf.Tensor]:
    """Defines the model serving signature."""
    context_spec, sequence_spec = pipeline.make_parsing_spec(self.model_config)
    context_input, sequence_input, _ = tf.io.parse_sequence_example(
        inputs, context_spec, sequence_spec
    )
    features = self.token_lookup_embedders.apply_string_lookup(
        context_input, sequence_input
    )
    predictions = self(features, training=False)

    outputs = {}
    outputs['context_embeddings'] = predictions[
        facade_loss.CONTEXT_EMBEDDINGS_KEY
    ]

    for action_name in sorted(self.action_towers.keys()):
      embeddings = predictions[action_name]
      action_embeddings_name = action_name + '_action_embeddings'
      outputs[action_embeddings_name] = embeddings.flat_values

      scores = predictions[action_name + SCORES_KEY_SUFFIX]
      action_scores_name = action_name + '_scores'
      outputs[action_scores_name] = scores.flat_values

    return outputs

  def train_step(
      self, minibatch: tuple[AnyMapping, AnyMapping]
  ) -> Mapping[str, tf.Tensor]:
    """Custom training step."""
    with tf.GradientTape() as tape:
      embeddings = self(minibatch, training=True)
      context_inputs, _ = minibatch
      embeddings[facade_loss.PRINCIPAL_IDENTITIES_KEY] = context_inputs[
          self.model_config.principal_feature_name
      ]
      loss = self.loss_layer(embeddings)

    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    # Do not invoke potentially costly metrics' result() at each batch step.
    return {'loss': loss}

  def add_evaluation_layers(self):
    """Adds layers and metrics for model evaluation purposes.

    Should be called before compile(), and only in the evaluator job(s) so as to
    not create unnecessary zero metrics for training job in tensorboard.
    """
    eval_config = self.model_config.training_hyperparameters.evaluation
    self.evaluation_contrastive_sampler = stateless.ContrastiveLabelsScores(
        self.model_config.scoring_function,
        eval_config.synthetic_positives.contrastive_scores_per_query,
        eval_config.synthetic_positives.positive_instances_weight_factor,
    )
    thresholds = list(eval_config.metrics_fpr_thresholds)
    self.evaluation_metrics = {}
    for name in sorted(self.action_towers):
      self.evaluation_metrics[name] = combined.ExactTruncatedMetrics(
          tpr_at_fpr_thresholds=thresholds,
          auc_at_fpr_thresholds=thresholds,
          prediction_at_fpr_thresholds=thresholds,
          name_prefix=name,
      )
    # Global (commensurate scores) metric.
    if ALL_ACTIONS_METRIC_NAME in self.evaluation_metrics:
      raise ValueError(
          'Action name collision with ALL_ACTIONS_METRIC_NAME:'
          f' {ALL_ACTIONS_METRIC_NAME}.'
      )
    self.evaluation_metrics[ALL_ACTIONS_METRIC_NAME] = (
        combined.ExactTruncatedMetrics(
            tpr_at_fpr_thresholds=thresholds,
            auc_at_fpr_thresholds=thresholds,
            prediction_at_fpr_thresholds=thresholds,
            name_prefix=ALL_ACTIONS_METRIC_NAME,
        )
    )

  def test_step(
      self, minibatch: tuple[AnyMapping, AnyMapping]
  ) -> Mapping[str, tf.Tensor]:
    """Custom evaluation step."""
    all_embeddings = self(minibatch, training=False)
    context_inputs, _ = minibatch
    principals = context_inputs[self.model_config.principal_feature_name]
    all_embeddings[facade_loss.PRINCIPAL_IDENTITIES_KEY] = principals
    loss = self.loss_layer(all_embeddings)  # Updates loss metrics.

    # Partially duplicates contrastive loss function logic to consistently
    # report performance on the same contrastive learning task.
    all_actions = []
    all_matching_items = []
    all_action_weights = []
    all_action_principals = []
    context_embeddings = all_embeddings[facade_loss.CONTEXT_EMBEDDINGS_KEY]

    assert isinstance(principals, tf.Tensor)
    # All principals should be hashed to nonnegative integers.
    check_has_principals = tf.debugging.assert_greater_equal(principals, 0)
    with tf.control_dependencies([check_has_principals]):
      principals = tf.squeeze(principals)

    for name in sorted(self.action_towers):
      ragged_embeddings = all_embeddings[name]
      action_embeddings = ragged_embeddings.flat_values
      all_actions.append(action_embeddings)
      action_principals = tf.repeat(principals, ragged_embeddings.row_lengths())
      all_action_principals.append(action_principals)

      matching_items = ragged_embeddings.value_rowids()
      all_matching_items.append(matching_items)

      action_weights = tf.repeat(
          self.model_config.training_hyperparameters.action_name_to_loss_weight[
              name
          ],
          tf.shape(action_embeddings)[0],
      )
      all_action_weights.append(action_weights)
      labels, scores, weights = self.evaluation_contrastive_sampler({
          'queries': action_embeddings,
          'items': context_embeddings,
          'matching_items': matching_items,
          'query_compatibility_keys': action_principals,
          'item_compatibility_keys': principals,
      })
      # Reduce memory requirements by dropping zero-weighted (spurious)
      # synthetic positives.
      labels = tf.boolean_mask(labels, weights > 0.0)
      scores = tf.boolean_mask(scores, weights > 0.0)
      weights = tf.boolean_mask(weights, weights > 0.0)
      self.evaluation_metrics[name].update_state(labels, scores, weights)

    # Global (commensurate scores) metric.
    labels, scores, weights = self.evaluation_contrastive_sampler({
        'queries': tf.concat(all_actions, axis=0),
        'items': context_embeddings,
        'matching_items': tf.concat(all_matching_items, axis=0),
        'query_weights': tf.concat(all_action_weights, axis=0),
        'item_weights': tf.ones(
            tf.shape(context_embeddings)[0], dtype=all_action_weights[0].dtype
        ),
        'queries_compatibility_keys': tf.concat(all_action_principals, axis=0),
        'items_compatibility_keys': principals,
    })
    labels = tf.boolean_mask(labels, weights > 0.0)
    scores = tf.boolean_mask(scores, weights > 0.0)
    weights = tf.boolean_mask(weights, weights > 0.0)
    self.evaluation_metrics[ALL_ACTIONS_METRIC_NAME].update_state(
        labels, scores, weights
    )

    # Do not invoke costly metrics' result() at each batch step.
    return {'loss': loss}

  # This method is called at the end of each training and evaluation epochs.
  # We override it here to compute every metrics' final results.
  def _validate_and_get_metrics_result(self, logs: Any) -> Mapping[str, Any]:
    logs.update(self.get_metrics_result())
    return super()._validate_and_get_metrics_result(logs)  # pytype: disable=attribute-error

  @property
  def metrics(self) -> Sequence[tf.keras.metrics.Metric]:
    # Caching the collection of metric ops does not invalidate correct metrics
    # tracking throuhout training and evaluation, but overcomes a performance
    # issue. See b/279489896.
    if self._cached_metrics is not None:
      return self._cached_metrics

    metrics = []
    if self._is_compiled:
      if self.compiled_loss is not None:
        metrics += self.compiled_loss.metrics
      if self.compiled_metrics is not None:
        metrics += self.compiled_metrics.metrics

    # Track evaluation metrics if present.
    if self.evaluation_metrics:
      for _, metric in sorted(self.evaluation_metrics.items()):
        metrics.append(metric)

    # Collect metrics in attributes of all sub-layers.
    for l in self._flatten_layers():
      metrics.extend(l._metrics)  # pylint: disable=protected-access

    self._cached_metrics = metrics
    logging.info('Collected model metrics: {}'.format(metrics))
    return metrics
