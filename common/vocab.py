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
"""Utility for extracting vocab tokens from SequenceExamples."""

import cityhash
import collections
import tensorflow as tf
from typing import List, Tuple

from protos import vocab_pb2

def _extract_tokens(example: tf.train.SequenceExample) -> List[Tuple[bool, str, bytes]]:
  """Extracts all bytes-list feature values from a SequenceExample."""
  # Extract tokens from the context features
  context_kvs = [
      (True, name, value)
      for name, feature in example.context.feature.items()
      if feature.HasField('bytes_list')
      for value in feature.bytes_list.value
  ]

  # Extract tokens from the sequence features
  sequence_kvs = [
      (False, name, value)
      for name, feature_list in example.feature_lists.feature_list.items()
      for feature in feature_list.feature
      if feature.HasField('bytes_list')
      for value in feature.bytes_list.value
  ]

  return context_kvs + sequence_kvs


def extract_vocabs(sequence_examples: List[tf.train.SequenceExample]) -> List[vocab_pb2.Vocab]:
  """Create the vocabs representing bytes values from the given SequenceExamples."""
  tokens_by_feature = collections.defaultdict(set)
  for example in sequence_examples:
    tokens = _extract_tokens(example)
    for token in tokens:
      tokens_by_feature[(token[0], token[1])].add(token[2])

  vocabs = []
  for k, v in tokens_by_feature.items():
    values = list(v)
    values.sort(key=cityhash.CityHash64)

    vocabs.append(vocab_pb2.Vocab(
        name = k[1],
        is_context = k[0],
        values = values
    ))
  return vocabs
