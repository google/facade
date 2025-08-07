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
"""Utility for converting Facade-internal data types to TFExamples."""

import tensorflow as tf
from typing import List, Tuple

from protos import contextualized_actions_pb2


PRINCIPAL_CONTEXT_FEATURE = 'p'
_DELIMITER = '/'


def get_base_tf_feature_name(source_type: str, feature_name: str) -> str:
  return source_type + _DELIMITER + feature_name


def get_tokens_tf_feature_name(base_name: str) -> str:
  return base_name + _DELIMITER + 't'


def get_weights_tf_feature_name(base_name: str) -> str:
  return base_name + _DELIMITER + 'w'


def make_tf_features(
    prefix: str, facade_feature: contextualized_actions_pb2.Feature
) -> List[Tuple[str, tf.train.Feature]]:
  """
  Translates a Facade Feature proto into a list of TF-Feature protos.
  """
  name_features: List[Tuple[str, tf.train.Feature]] = []
  name = get_base_tf_feature_name(prefix, facade_feature.name)
    
  feature_type = facade_feature.WhichOneof('type')

  # if feature_type == 'bag_of_weighted_words':
  tokens_feature_name = get_tokens_tf_feature_name(name)
  tokens_feature = tf.train.Feature()        
  weights_feature_name = get_weights_tf_feature_name(name)
  weights_feature = tf.train.Feature()
  tokens_feature.bytes_list.Clear()
  weights_feature.float_list.Clear()
  for wt in facade_feature.bag_of_weighted_words.tokens:
    # NOTE: Python strings must be encoded to bytes for a BytesList.
    tokens_feature.bytes_list.value.append(wt.token)
    weights_feature.float_list.value.append(wt.weight)
  name_features.append((tokens_feature_name, tokens_feature))
  name_features.append((weights_feature_name, weights_feature))

  return name_features


def to_tf_input(
    contextualized_actions: contextualized_actions_pb2.ContextualizedActions
) -> tf.train.SequenceExample:
  """
  Serializes ContextualizedActions into a tensorflow input format the model
  can operate on. All actions within the same source type must specify the
  exact same features, with possibly empty values. Performs lightweight and
  incomplete sanity checks.
  """
  out = tf.train.SequenceExample()

  # Set context features
  context_features = out.context.feature
  context_features[PRINCIPAL_CONTEXT_FEATURE].bytes_list.value.append(
      contextualized_actions.principal.encode('utf-8')
  )

  for csf in contextualized_actions.context.features_per_source:
    prefix = csf.source_type
    for cf in csf.features:
      for name, feature in make_tf_features(prefix, cf):
        if name in context_features:
          raise ValueError(
              f"Duplicate TF feature name: '{name}' in context features."
          )
        context_features[name].CopyFrom(feature)

  # Set action features
  action_feature_lists = out.feature_lists.feature_list
  for fabs in contextualized_actions.actions:
    prefix = fabs.source_type
    num_actions = 0
    for fa in fabs.actions:
      # Process all features for a single action (one step in the sequence)
      for f in fa.features:
        for name, feature in make_tf_features(prefix, f):
          # Get the list for this feature name (creates it if new)
          feature_list = action_feature_lists[name]
                    
          # This check ensures that all actions have the same features.
          # Before adding the feature for the current action, the list's
          # length must equal the number of actions processed so far.
          if len(feature_list.feature) != num_actions:
            raise ValueError(
                f"Missing or extra feature '{name}' for action of "
                f"type '{prefix}'. Inconsistent feature list lengths."
            )
                    
          # Add the new feature to the end of its list.
          feature_list.feature.add().CopyFrom(feature)
            
      num_actions += 1

  return out
