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
"""Configures the base token embedders with possibly shared weights."""

import collections
from typing import Container, Mapping, Sequence
import glob

from absl import logging
import cityhash
import tensorflow as tf
from tensorflow import keras

from protos import config_pb2
from protos import vocab_pb2
from model.common import write_once_sorted_dict as wosd
from model.layers import oov_dropout


SetMapping = Mapping[str, set[str]]
AnyTensor = tf.Tensor | tf.SparseTensor | tf.RaggedTensor
AnyMapping = Mapping[str, AnyTensor]
LookupEmbedderMapping = Mapping[str, '_TokenLookupEmbedder']


@tf.keras.utils.register_keras_serializable(package='Facade')
class _TokenLookupEmbedder(tf.keras.layers.Layer):
  """Looks up and embeds a single vocabularies of tokens."""

  def __init__(
      self,
      vocabulary: set[bytes],
      oov_dropout_rate: float,
      num_oov_indices: int,
      dimensions: int,
      feature_names: Sequence[str],
      embedding_name: str,
  ):
    """Initializes the lookup and embedder layer.

    Args:
      vocabulary: The set of tokens to be embedded.
      oov_dropout_rate: The rate at which to replace tokens with OOV tokens
        during training.
      num_oov_indices: The number of OOV indices to use.
      dimensions: The number of embedding dimensions. If non-positive, the
        number of dimensions is set to 6 * sqrt(vocabulary_size + num_oov).
      feature_names: The names of the TF Features to be looked up and embedded.
      embedding_name: The name of the embedding.
    """
    super().__init__(name=f'TokenLookupEmbedder_{embedding_name}')
    vocabulary = list(vocabulary)
    # Deterministically shuffle the tokens.
    vocabulary.sort(key=cityhash.CityHash64)
    num_oov_indices = max(1, num_oov_indices)
    if dimensions <= 0:
      dimensions = int(6 * (len(vocabulary) + num_oov_indices) ** 0.25)

    self._lookup = keras.layers.StringLookup(
        num_oov_indices=num_oov_indices, vocabulary=vocabulary
    )

    self._oov_dropout = oov_dropout.OovDropout(
        dropout_prob=oov_dropout_rate, num_oov_tokens=num_oov_indices
    )

    self._embed = keras.layers.Embedding(
        input_dim=len(vocabulary) + num_oov_indices,
        output_dim=dimensions,
        embeddings_initializer='TruncatedNormal',
    )

    self._features = feature_names

    logging.info(
        'Token embedding name `%s`:\n\t`%s` tokens\n\t`%s` dimensions\n\t'
        'covers TF features: `%s`\n\tsample tokens: `%s`',
        embedding_name,
        len(vocabulary),
        dimensions,
        feature_names,
        list(vocabulary)[:10],
    )

  def lookup_features(self, inputs: AnyMapping, training: bool) -> AnyMapping:
    outputs = dict(inputs)
    for feature_name in self._features:
      if feature_name not in inputs:
        continue
      string_tokens = inputs[feature_name]
      outputs[feature_name] = self._oov_dropout(
          self._lookup(string_tokens), training=training
      )

    return outputs

  def embed_features(self, inputs: AnyMapping) -> AnyMapping:
    outputs = dict(inputs)
    for feature_name in self._features:
      if feature_name not in inputs:
        continue
      outputs[feature_name] = self._embed(inputs[feature_name])

    return outputs

  def call(self, *args, **kwargs):
    raise ValueError('TokenLookupEmbedder is not a callable layer.')


def _read_vocabulary(
    filepattern: str,
) -> tuple[SetMapping, SetMapping]:
  """Reads vocabularies from TFRecord files of `Vocab` protos.

  Args:
    filepattern: Glob pattern for the TFRecord vocabulary file(s).

  Returns:
    Two dictionaries of vocabularies, mapping feature names to a set of
    unique tokens. The first dictionary is for context features, and the second
    is for sequence features.
  """
  context_vocabs = wosd.WriteOnceSortedDict()
  sequence_vocabs = wosd.WriteOnceSortedDict()
  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(filepattern))
  for raw_record in dataset:
    vocab = vocab_pb2.Vocab()
    vocab.ParseFromString(raw_record.numpy())
    vocabulary = set(vocab.values)
    if vocab.is_context:
      context_vocabs[vocab.name] = vocabulary
      logging.info(
        'Read vocabulary for context feature `%s`: %i tokens.',
        vocab.name,
        len(vocabulary),
      )
    else:
      sequence_vocabs[vocab.name] = vocabulary
      logging.info(
        'Read vocabulary for sequence feature `%s`: %i tokens.',
        vocab.name,
        len(vocabulary),
      )

  return context_vocabs, sequence_vocabs


def _collect_embedding_feature_names(
    architecture: config_pb2.Architecture,
    features_with_vocabularies: Container[str],
) -> SetMapping:
  """Collects TF Feature name-embedding name associations in a SegmentEmbedder.

  This also checks that the TF Feature names have a corresponding vocabulary.

  Args:
    architecture: The architecture of either an action tower, or the context
      tower.
    features_with_vocabularies: TF Feature names for which vocabularies are
      present.

  Returns:
    For each token_embedding_name the set of associated TF Feature names.

  Raises:
    ValueError: if the model configuration defines a TF Feature name which does
      not have an associated vocabulary.
  """
  match architecture.WhichOneof('architecture'):
    case 'concatenate_then_snn':
      embeddings_features = collections.defaultdict(set)
      for (
          segment_reduction
      ) in architecture.concatenate_then_snn.segment_reductions:
        feature_name = segment_reduction.token_feature_name
        if feature_name not in features_with_vocabularies:
          raise ValueError(
              'Model configuration defines feature name'
              f' {feature_name} but no corresponding vocabulary was found'
          )
        embeddings_features[segment_reduction.token_embedding_name].add(
            feature_name
        )
      return embeddings_features

    case _:
      raise ValueError(f'Unknown architecture {architecture}.')


class TokenLookupEmbedders(tf.Module):
  """Holds the base token lookup and embedders with possibly shared weights."""

  def __init__(
      self,
      vocabulary_filepattern: str,
      model_config: config_pb2.ModelHyperparameters,
      name: str | None = None,
  ):
    """Constructs the possibly shared lookup and embedders modules.

    Args:
      vocabulary_filepattern: Creates embedding vocabularies based on the token
        vocabulary files matching this pattern. Vocabulary files contain the set
        of all distinct tokens per TF Feature name.
      model_config: The model configuration which contains the embeddings
        configuration with weight sharing specifications.
      name: A name for this module.

    Raises:
      ValueError if the model configuration defines a feature name that is
        absent from the vocabulary file(s), or if the model architecture is not
        supported.
    """
    super().__init__(name=name)
    context_vocab, sequence_vocab = _read_vocabulary(vocabulary_filepattern)

    embedding_features = _collect_embedding_feature_names(
        model_config.context_architecture, context_vocab.keys()
    )

    for architecture in model_config.action_name_to_architecture.values():
      for embedding_name, feature_names in _collect_embedding_feature_names(
          architecture, sequence_vocab.keys()
      ).items():
        embedding_features[embedding_name].update(feature_names)

    self.embedding_name_to_lookup_embedder_features = {}
    # Construct the embedders in deterministic order.
    for embedding_name, feature_names in sorted(embedding_features.items()):
      if embedding_name not in model_config.token_embedding_name_to_config:
        raise ValueError(
            f'Missing configuration for token_embedding_name {embedding_name}.'
        )
      # Merge vocabularies.
      vocabulary = set.union(
          *(
              context_vocab[feature_name]
              for feature_name in feature_names
              if feature_name in context_vocab
          ),
          *(
              sequence_vocab[feature_name]
              for feature_name in feature_names
              if feature_name in sequence_vocab
          ),
      )
      config = model_config.token_embedding_name_to_config[embedding_name]
      self.embedding_name_to_lookup_embedder_features[embedding_name] = (
          _TokenLookupEmbedder(
              vocabulary,
              config.oov_dropout_rate,
              config.num_oov_indices,
              config.dimensions,
              feature_names,
              embedding_name,
          )
      )

  def apply_string_lookup(
      self,
      context_inputs: AnyMapping,
      sequence_inputs: AnyMapping,
      training: bool = False,
  ) -> tuple[AnyMapping, AnyMapping]:
    """Apply string lookup layers to integerize every string input."""
    for (
        lookup_embedder
    ) in self.embedding_name_to_lookup_embedder_features.values():
      context_inputs = lookup_embedder.lookup_features(context_inputs, training)
      sequence_inputs = lookup_embedder.lookup_features(
          sequence_inputs, training
      )

    return context_inputs, sequence_inputs

  def apply_embedders(
      self,
      context_inputs: AnyMapping,
      sequence_inputs: AnyMapping,
  ) -> tuple[AnyMapping, AnyMapping]:
    """Apply embedders to context and sequence mappings."""
    for (
        lookup_embedder
    ) in self.embedding_name_to_lookup_embedder_features.values():
      context_inputs = lookup_embedder.embed_features(context_inputs)
      sequence_inputs = lookup_embedder.embed_features(sequence_inputs)
    return context_inputs, sequence_inputs
