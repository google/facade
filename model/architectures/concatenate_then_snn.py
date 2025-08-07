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
"""The classic Facade tower architecture."""

from typing import Any, Mapping, Optional, Union

import tensorflow as tf
from tensorflow import keras

from google.protobuf import json_format
from model.layers import dense_importer
from model.layers import segment_embedder
from model.layers import snn
from model.layers import stateless
from protos import config_pb2


class ConcatenateThenSNNTower(keras.layers.Layer):
  """One tower (context or action): create embeddings, concat, then SNN."""

  def __init__(
      self,
      model_config: config_pb2.ModelHyperparameters,
      action_name: Optional[str] = None,
  ):
    """Constructs the layer.

    Args:
      model_config: The model configuration which exclusively contains
        ConcatenateThenSNN architectures.
      action_name: If none, this constructs a tower for the context features. If
        provided, constructs a tower for a single action type. Note that context
        towers return a fixed-size dense tf.Tensor of context embeddings, and
        action towers return a ragged tensor of action embeddings.
    """
    super().__init__()
    self._model_config = model_config
    self._action_name = action_name
    if action_name is None:
      architecture = model_config.context_architecture
      transformations = model_config.context_embeddings_transformations
    else:
      if action_name not in model_config.action_name_to_architecture:
        raise ValueError(
            f'No architecture for action name `{action_name}` provided in model'
            ' configuration.'
        )
      architecture = model_config.action_name_to_architecture[action_name]
      transformations = model_config.action_embeddings_transformations

    if not architecture.HasField('concatenate_then_snn'):
      raise ValueError(
          f'Expected concatenate_then_snn architecture, got: {architecture}.'
      )
    architecture = architecture.concatenate_then_snn

    self._segment_embedder = {}
    for reduction_conf in architecture.segment_reductions:
      self._segment_embedder[reduction_conf.token_feature_name] = (
          reduction_conf.intensity_feature_name,
          segment_embedder.SegmentEmbedder(
              reduction_conf.segment_weight_scaling,
              reduction_conf.segment_weight_normalization,
              model_config.training_hyperparameters.dropout_tokens,
          ),
      )
    for dense_conf in architecture.fixed_size_dense_features:
      self._segment_embedder[dense_conf.feature_name] = (
          '',  # no intensities
          dense_importer.DenseImporter(dense_conf.size),
      )
    self._snn = snn.SNN(
        list(architecture.snn.layer_sizes) + [model_config.embedding_dims],
        model_config.training_hyperparameters.dropout_neurons,
    )
    self._transformation = stateless.EmbeddingsTransformation(transformations)

  def get_config(self) -> Mapping[str, Any]:
    return {
        'model_config': json_format.MessageToJson(self._model_config),
        'action_name': self._action_name,
    }

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> 'ConcatenateThenSNNTower':
    model_config = config_pb2.ModelHyperparameters()
    json_format.Parse(config['model_config'], model_config)
    return cls(model_config, config['action_name'])

  def call(
      self,
      inputs: Mapping[str, Union[tf.Tensor, tf.SparseTensor]],
      training: Optional[bool] = None,
  ) -> Union[tf.Tensor, tf.RaggedTensor]:
    """Applies the layer."""
    segment_embeddings = []
    for feature_name, (
        intensity_feature_name,
        segment_embed,
    ) in sorted(self._segment_embedder.items()):
      segment_args = {}

      if isinstance(segment_embed, segment_embedder.SegmentEmbedder):
        segment_args['embeddings'] = inputs[feature_name]
        if intensity_feature_name:
          segment_args['intensities'] = inputs[intensity_feature_name]
      elif isinstance(segment_embed, dense_importer.DenseImporter):
        segment_args['dense_vector'] = inputs[feature_name]
      else:
        raise ValueError(f'Unknown segment_embed {segment_embed}.')

      segment_embeddings.append(
          segment_embed(
              segment_args,
              training=training,
          )
      )

    output = tf.concat(segment_embeddings, axis=-1)
    flat_values = output if self._action_name is None else output.flat_values
    flat_values = self._snn(flat_values, training=training)
    flat_values = self._transformation(flat_values)
    if self._action_name is None:
      return flat_values
    else:
      return output.with_flat_values(flat_values)
