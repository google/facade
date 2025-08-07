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
"""Defines the overall training loss function used by a Facade model."""

from collections.abc import Mapping
from typing import Any, Union

import tensorflow as tf
from tensorflow import keras

from google.protobuf import json_format
from model.layers import stateless
from model.loss import generalized_logistic
from model.loss import multi_similarity
from model.loss import pairwise_huber
from model.loss import rescale
from model.loss import sanll
from protos import config_pb2


CONTEXT_EMBEDDINGS_KEY = '$context_internal'
PRINCIPAL_IDENTITIES_KEY = '$principal_internal'

_EPSILON = 1e-6


class FacadeModelLoss(keras.layers.Layer):
  """Defines the overall training loss function used by a Facade model.

  The only currently supported strategy is contrastive-based with a pairwise
  Huber loss function.
  """

  def __init__(self, model_config: config_pb2.ModelHyperparameters):
    """Creates the loss layer."""
    super().__init__()
    self.model_config = model_config
    training_config = model_config.training_hyperparameters
    self.mean_global_loss = keras.metrics.Mean('mean_global_loss')
    if not training_config.commensurable_scores_across_actions:
      self.mean_action_loss = {}
      for name in sorted(model_config.action_name_to_architecture.keys()):
        metric = keras.metrics.Mean(f'{name}/mean_action_loss')
        self.mean_action_loss[name] = metric
        # Store a reference to the Metric object as a layer attribute for
        # proper Metrics collection from the top-level Model object.
        setattr(self, f'{name}_mean_action_loss', metric)

    if not self.model_config.action_name_to_architecture:
      raise ValueError(
          'Empty action_name_to_architecture, expected at least one action.'
      )

    # Configure optional rescalers.
    rescalers = []
    if training_config.loss_function.log_rescaler:
      rescalers.append(
          keras.layers.Lambda(lambda x: -tf.math.log(_EPSILON + 1.0 - x))
      )
    if training_config.loss_function.HasField('linear_rescaler'):
      rescalers.append(
          rescale.LinearRescaler(training_config.loss_function.linear_rescaler)
      )
    if not rescalers:
      rescalers.append(keras.layers.Identity())
    self.rescaler = keras.Sequential(rescalers)

    # Configure the loss function.
    loss_function_config = training_config.loss_function
    match loss_function_config.WhichOneof('loss_function'):
      case 'pairwise_huber':
        params = loss_function_config.pairwise_huber
        self.loss_function = pairwise_huber.PairwiseHuberLoss(
            params.soft_margin,
            params.hard_margin,
            params.norm_push,
            params.lse_scale,
        )
      case 'generalized_logistic':
        params = loss_function_config.generalized_logistic
        self.loss_function = generalized_logistic.GeneralizedLogisticLoss(
            params.soft_margin, params.hard_margin, params.negative_push
        )
      case 'sanll':
        params = loss_function_config.sanll
        self.loss_function = sanll.SanllLoss(
            params.margin, params.negative_push
        )
      case 'multi_similarity':
        params = loss_function_config.multi_similarity
        self.loss_function = multi_similarity.MultiSimilarityLoss(
            params.a, params.b, params.loc
        )
      case _:
        raise ValueError(f'Unknown loss function {loss_function_config}.')

    # Configure the synthetic positive training strategy.
    strategy = training_config.synthetic_positives_strategy
    match strategy.WhichOneof('strategy'):
      case 'random_sample_within_minibatch':
        random_sample_config = strategy.random_sample_within_minibatch
        contrastive_scores = random_sample_config.contrastive_scores_per_query
        weight_factor = random_sample_config.positive_instances_weight_factor
        if contrastive_scores <= 0:
          raise ValueError('Expected positive contrastive_scores_per_query.')
        self.contrastive_sampler = stateless.ContrastiveLabelsScores(
            model_config.scoring_function,
            contrastive_scores,
            weight_factor,
        )
      case _:
        raise ValueError(f'Unknown synthetic positives strategy {strategy}.')

  def get_config(self) -> Mapping[str, Any]:
    return {'model_config': json_format.MessageToJson(self.model_config)}

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> 'FacadeModelLoss':
    model_config = config_pb2.ModelHyperparameters()
    json_format.Parse(config['model_config'], model_config)
    return cls(model_config)

  def call(
      self,
      inputs: Mapping[str, Union[tf.Tensor, tf.RaggedTensor]],
  ) -> tf.Tensor:
    """Makes the loss graph.

    Args:
      inputs: Named embeddings, containing both the context and at all action
        embeddings. The context embeddings are a tf.Tensor keyed by
        CONTEXT_EMBEDDINGS_KEY. The principals' identities corresponding to the
        contexts are a tf.Tensor keyed by PRINCIPAL_IDENTITIES_KEY. These
        identities are used to prune spurious synthetic positive crossings. The
        action embeddings are tf.RaggedTensors.

    Returns:
      A scalar loss suitable for model training by gradient descent.
    """
    training_config = self.model_config.training_hyperparameters
    context_embeddings = inputs[CONTEXT_EMBEDDINGS_KEY]
    assert isinstance(context_embeddings, tf.Tensor)

    principals = inputs[PRINCIPAL_IDENTITIES_KEY]
    assert isinstance(principals, tf.Tensor)
    principals = tf.squeeze(principals)

    if not training_config.commensurable_scores_across_actions:
      losses = []
      for name in sorted(self.model_config.action_name_to_architecture.keys()):
        ragged_embeddings = inputs[name]
        embeddings = ragged_embeddings.flat_values
        matching_items = ragged_embeddings.value_rowids()
        action_principals = tf.repeat(
            principals, ragged_embeddings.row_lengths()
        )
        # `weights`` acts as a mask here, where spurious synthetic positives
        # are weighted zero.
        labels, scores, weights = self.contrastive_sampler({
            'queries': embeddings,
            'items': context_embeddings,
            'matching_items': matching_items,
            'query_compatibility_keys': action_principals,
            'item_compatibility_keys': principals,
        })
        scores = self.rescaler(scores)
        if isinstance(
            self.loss_function, pairwise_huber.PairwiseHuberLoss
        ) or isinstance(
            self.loss_function, multi_similarity.MultiSimilarityLoss
        ):
          action_loss = self.loss_function((labels, scores, weights))
        else:
          action_loss = self.loss_function(labels, scores, weights)
        action_loss = tf.cast(action_loss, tf.float32)
        self.mean_action_loss[name].update_state(action_loss)
        if name not in training_config.action_name_to_loss_weight:
          raise ValueError(f'Missing loss weight for action "{name}".')
        # This is the global weight for the action type.
        weight = training_config.action_name_to_loss_weight[name]
        losses.append(weight * action_loss)
      loss = tf.add_n(losses)
      self.mean_global_loss.update_state(loss)
      return loss

    # Commensurable scores.
    all_queries = []
    all_matching_items = []
    all_weights = []
    all_action_principals = []
    for name in sorted(self.model_config.action_name_to_architecture.keys()):
      ragged_embeddings = inputs[name]
      queries = ragged_embeddings.flat_values
      all_queries.append(queries)
      all_matching_items.append(ragged_embeddings.value_rowids())
      action_principals = tf.repeat(principals, ragged_embeddings.row_lengths())
      all_action_principals.append(action_principals)
      if name not in training_config.action_name_to_loss_weight:
        raise ValueError(f'Missing loss weight for action {name}.')
      weight = training_config.action_name_to_loss_weight[name]
      all_weights.append(tf.repeat(weight, tf.shape(queries)[0]))

    labels, scores, weights = self.contrastive_sampler({
        'queries': tf.concat(all_queries, axis=0),
        'items': context_embeddings,
        'matching_items': tf.concat(all_matching_items, axis=0),
        'query_weights': tf.concat(all_weights, axis=0),
        'item_weights': tf.ones(
            tf.shape(context_embeddings)[0], dtype=all_weights[0].dtype
        ),
        'query_compatibility_keys': tf.concat(all_action_principals, axis=0),
        'item_compatibility_keys': principals,
    })
    scores = self.rescaler(scores)
    if isinstance(
        self.loss_function, pairwise_huber.PairwiseHuberLoss
    ) or isinstance(self.loss_function, multi_similarity.MultiSimilarityLoss):
      loss = self.loss_function((labels, scores, weights))
    else:
      loss = self.loss_function(labels, scores, weights)
    self.mean_global_loss.update_state(loss)
    return loss
