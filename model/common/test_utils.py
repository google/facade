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
"""Utilities for writing tests that rely on vocabulary files."""

import tempfile
from typing import Mapping, List, Tuple

import tensorflow as tf
from tensorflow.core.example import example_pb2
from google.protobuf import text_format

from protos import vocab_pb2


def write_vocabulary_file(
    feature_name_to_vocabulary: Mapping[str, Tuple[bool, List[bytes]]],
) -> str:
  """Makes a TFRecord vocabulary file for testing."""
  fp = tempfile.NamedTemporaryFile('wb', delete=False)
  filename = fp.name
  fp.close()

  with tf.io.TFRecordWriter(filename) as writer:
    for feature_name, (is_context, values) in feature_name_to_vocabulary.items():
      vocab_proto = vocab_pb2.Vocab(
          name=feature_name,
          values=values,
          is_context=is_context
      )
      writer.write(vocab_proto.SerializeToString())
  return filename


def write_example_file(textproto_examples: str) -> str:
  """Takes a file of TFSequenceExample textprotos and makes a TFRecord
  file of them for testing, returning the filename."""
  fp = tempfile.NamedTemporaryFile('wb', delete=False)
  filename = fp.name
  fp.close()

  with tf.io.TFRecordWriter(filename) as writer:
    with open(textproto_examples, "r") as reader:
      txt = ''
      for line in reader:
        if line == '\n':
          writer.write(text_format.Parse(txt, example_pb2.SequenceExample()).SerializeToString())
          txt = ''
        else:
          txt += line
      writer.write(text_format.Parse(txt, example_pb2.SequenceExample()).SerializeToString())
  return filename
