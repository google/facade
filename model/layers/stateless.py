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
"""Stateless layers.

A stateless layer does not own tf.Variable(s) and thus does not maintain state
between iterations.
"""

from collections.abc import Sequence
from typing import Any, Union

import tensorflow as tf
from tensorflow import keras

from protos import config_pb2

_EPSILON = 1e-6


class EmbeddingsTransformation(keras.layers.Layer):
  """Stateless sequential transformations of embeddings.

  Expects rank 2 tensor inputs, where the embeddings dimension is last.
  """

  def __init__(self, transformations: Sequence['config_pb2.Transformation']):
    """Creates a transformation layer.

    Args:
      transformations: sequence of transformations to apply.
    """
    super().__init__(trainable=False)
    self._transformations = transformations
    tr = config_pb2.Transformation
    transform_options = {
        tr.TR_IDENTITY: tf.identity,
        tr.TR_SIGMOID: tf.math.sigmoid,
        tr.TR_SOFTPLUS: tf.math.softplus,
        # Assumes inputs are rank-2 and last dimension is embeddings, take
        # normalization operations on that dimension.
        tr.TR_SOFTMAX: lambda x: tf.nn.softmax(x, axis=1),
        tr.TR_L2_NORMALIZED: lambda x: tf.math.l2_normalize(x, axis=1),
    }
    self._ops = []
    for transformation in transformations:
      self._ops.append(transform_options[transformation])

  def get_config(self) -> dict[str, Any]:
    return {'transformations': list(self._transformations)}

  def call(self, x: tf.Tensor) -> tf.Tensor:
    for op in self._ops:
      x = op(x)
    return x


class EmbeddingsScorer(keras.layers.Layer):
  """Creates scores from context and action embeddings."""

  def __init__(self, scoring_function: config_pb2.ScoringFunction):
    super().__init__(trainable=False)
    self._scoring_function = scoring_function

  def get_config(self) -> dict[str, Any]:
    return {'scoring_function': self._scoring_function}

  def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Computes the action-context compatibility scores.

    Args:
      inputs: A tuple of rank 2 tensors holding the action and context
        embeddings respectively. Tensor shapes must match exactly.

    Returns:
      scores: A rank 1 tensor of floats holding the scores.

    Raises:
      ValueError if unknown scoring function type.
    """
    actions, contexts = inputs
    if self._scoring_function == config_pb2.ScoringFunction.SF_DOT:
      return tf.einsum('be,be->b', actions, contexts)
    elif self._scoring_function == config_pb2.ScoringFunction.SF_OMDOT:
      return 1.0 - tf.einsum('be,be->b', actions, contexts)
    elif self._scoring_function == config_pb2.ScoringFunction.SF_HARDMIN:
      return tf.math.reduce_sum(tf.math.minimum(actions, contexts), axis=1)
    elif self._scoring_function == config_pb2.ScoringFunction.SF_SOFTMIN:
      contexts = contexts / (actions + contexts + _EPSILON)
      return tf.einsum('be,be->b', actions, contexts)
    else:
      raise ValueError(f'Unknown scoring function {self._scoring_function}')


class ContrastiveLabelsScores(keras.layers.Layer):
  """Generates negative and positive scores for contrastive learning.

  This layer generates negative (matching pairs) scores under the assumption
  that given `queries` and `items` embeddings, there is exactly one matching
  item per query. The per-query matching items indices are stored into
  `matching_items`.
  To generate the positive (synthetic pairs) scores, this layer randomly matches
  query-item pairs together. For random pairs of matching elements, or
  alternatively, if the caller-specified `compatibility_keys` of the elements
  are equal, the corresponding output weight is set to zero.
  """

  def __init__(
      self,
      scoring_function: config_pb2.ScoringFunction,
      contrastive_scores_per_query: int,
      positive_instances_weight_factor: float,
  ):
    """Constructs the layer.

    Args:
      scoring_function: The scoring function to use when computing query-item
        scores.
      contrastive_scores_per_query: This layer generates exactly
        `contrastive_scores_per_query * |queries|` randomly matched synthetic
        positive pairs, some of which are spurious (matching elements). For
        those pairs, the corresponding weight is zero.
      positive_instances_weight_factor: Multiplicative weight scaling factor for
        the positive (synthetic/contrastive) instance weights. Must be positive.

    Raises:
      ValueError if contrastive_scores_per_query is negative.
    """
    super().__init__(trainable=False)
    if contrastive_scores_per_query < 0:
      raise ValueError('contrastive_scores_per_query must be non-negative.')
    if positive_instances_weight_factor <= 0:
      raise ValueError(
          'positive_instances_weight_factor must be positive. Got:'
          f' {positive_instances_weight_factor}'
      )
    self._scoring_function = scoring_function
    self._contrastive_scores_per_query = contrastive_scores_per_query
    self._positive_instances_weight_factor = positive_instances_weight_factor
    self._scorer = EmbeddingsScorer(scoring_function)

  def get_config(self) -> dict[str, Any]:
    return {
        'scoring_function': self._scoring_function,
        'contrastive_scores_per_query': self._contrastive_scores_per_query,
        'positive_instances_weight_factor': (
            self._positive_instances_weight_factor
        ),
    }

  def call(
      self,
      inputs: dict[str, tf.Tensor],
  ) -> Union[
      tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor]
  ]:
    """Applies the layer.

    Args:
      inputs: A dictionary of tensors contains the following entries.
        * 'queries':  A rank 2 float tensor with shape [Q, E] holding the
          embeddings of Q queries. May be empty.
        * 'items': A rank 2 float tensor with shape [I, E] holding the
          embeddings of I items. Assumed non-empty.
        * 'matching_items': A rank 1 int tensor with shape [Q] holding the index
          of the matching item for every query.
        * [Optional] 'query_weights': A rank 1 float tensor with shape [Q]
          holding per-query weights. Defaults to unitary weights.
        * [Optional] 'item_weights': A rank 1 float tensor with shape [I]
          holding per-item weights. Defaults to unitary weights.
        * [Optional] 'query_compatibility_keys': A rank 1 tensor with shape [Q]
          of any type supporting equality testing. If not provided, defaults to
          `matching_items`.
        * [Optional] 'item_compatibility_keys': A rank 1 tensor with shape [I]
          with same type as `queries_compatibility_keys`. If not provided,
          defaults to using the row indices of `items`.

    Returns:
      labels: A rank 1 float {-1; +1} valued tensor representing the labels.
      scores: A rank 1 float tensor holding the scores.
      weights: A rank 1 float tensor representing the instance weights. The
        weights of synthetic positive instances where elements match (spurious
        instances) is set to zero. All tensors share the same shape.

    Raises:
      ValueError: if either only one of (query|item)_compatibility_keys or
      only one of (query|item)_weights is specified.
    """  # fmt: skip
    queries = inputs['queries']
    items = inputs['items']
    matching_items = inputs['matching_items']

    query_keys = inputs.get('query_compatibility_keys')
    item_keys = inputs.get('item_compatibility_keys')
    if (query_keys is None) ^ (item_keys is None):
      raise ValueError(
          '(query|item)_compatibility_keys must be simultaneously specified.'
      )
    if query_keys is None:
      assert item_keys is None
      query_keys = matching_items
      item_keys = tf.range(0, tf.shape(items)[0], dtype=matching_items.dtype)

    query_weights = inputs.get('query_weights')
    item_weights = inputs.get('item_weights')
    if (query_weights is None) ^ (item_weights is None):
      raise ValueError('(query|item)_weights must be simulatenously specified.')
    explicit_weights = query_weights is not None

    # Negative instances (matching items).
    neg_scores = self._scorer((queries, tf.gather(items, matching_items)))
    if explicit_weights:
      neg_weights = query_weights * tf.gather(item_weights, matching_items)
    else:
      neg_weights = tf.ones_like(neg_scores)

    all_scores = [neg_scores]
    all_weights = [neg_weights]
    all_labels = [-tf.ones_like(neg_scores)]

    # Positive instances by random pairing strategy. This minimizes tf.gather()
    # (data copies) by only shuffling the items. The random sampling strategy
    # samples more evenly distributed indices, as in having less repetitions,
    # than the straightforward tf.random.uniform strategy.
    n_items = tf.shape(items)[0]
    n_queries = tf.shape(queries)[0]
    base_ix = tf.tile(
        tf.range(0, n_items, dtype=tf.int32),
        [tf.cast(tf.math.ceil(n_queries / n_items), tf.int32)],
    )
    for _ in range(self._contrastive_scores_per_query):
      ix = tf.random.shuffle(base_ix)
      ix = ix[:n_queries]
      scores = self._scorer((queries, tf.gather(items, ix)))
      all_scores.append(scores)
      all_labels.append(tf.ones_like(scores))
      # Mark spurious synthetics.
      allowed = tf.not_equal(query_keys, tf.gather(item_keys, ix))
      if explicit_weights:
        weights = query_weights * tf.gather(item_weights, ix)
        weights = weights * tf.cast(allowed, weights.dtype)
      else:
        weights = tf.cast(allowed, tf.float32)
      weights *= self._positive_instances_weight_factor
      all_weights.append(weights)

    labels = tf.concat(all_labels, axis=0)
    scores = tf.concat(all_scores, axis=0)
    weights = tf.concat(all_weights, axis=0)
    return labels, scores, weights
