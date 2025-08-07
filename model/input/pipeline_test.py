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
import os
from typing import Any

from absl import flags
import numpy as np
import tensorflow as tf

from model.common import configuration
from model.common import test_utils
from model.input import pipeline as input_pipeline
from protos import config_pb2


AnyTensor = tf.Tensor | tf.SparseTensor | tf.RaggedTensor


# 'person3' in 'p' and 'NYC' in cost_cneter are absent in the vocab file.
# They will automatically get index 0.
FEATURE_NAME_TO_VOCABULARY = {
    'hr/cost_center': (
        True,
        [
            b'SYD',
            b'ORD',
            b'CDG',
        ],
    ),
    'code/reviewer_f/t': (
        True,
        [
            b'reviewer2',
            b'reviewer1',
            b'reviewer8',
            b'reviewer3',
        ],
    ),
    'drive/title': (
        False,
        [
            b'this',
            b'other',
            b'doc',
        ],
    ),
    'drive/owner': (
        False,
        [
            b'person2',
            b'person1',
        ],
    ),
    'network/prin/t': (
        False,
        [
            b'up2',
            b'up1',
            b'up3',
            b'up4',
            b'up5',
        ],
    ),
}

VOCABULARY_FILE = ''


def setUpModule():
  global VOCABULARY_FILE
  VOCABULARY_FILE = test_utils.write_vocabulary_file(
      FEATURE_NAME_TO_VOCABULARY
  )


class InputPipelineTest(tf.test.TestCase):
  def setUp(self):
    super(InputPipelineTest, self).setUp()
    self.config = configuration.read_config('model/input/test_config.textproto')
    self.record_filename = test_utils.write_example_file('model/input/test_examples.textproto')

  def test_input_pipeline_loops(self):
    input_fn = input_pipeline.make_from_disk_input_fn(
        self.record_filename, self.config, VOCABULARY_FILE, batch_size=5
    )
    input_context = tf.distribute.InputContext()
    dataset = input_fn(input_context)
    min_iterations = 100
    for iteration, _ in enumerate(dataset):
      if iteration > min_iterations:
        return
    self.fail(f'Expected at least {min_iterations} batches.')

  def assertDenseValues(
      self,
      x: AnyTensor,
      expected_type: tf.DType,
      expected_values: list[Any],
  ):
    self.assertIsInstance(x, tf.Tensor)
    self.assertEqual(x.dtype, expected_type)
    x_values = np.concatenate(x.numpy(), axis=None)
    if x_values.dtype.kind == 'f':  # floating point
      x_values = [float('{:.2f}'.format(i)) for i in x_values]
    self.assertAllEqual(
        x_values,
        expected_values,
    )

  def assertRaggedValues(
      self,
      x: AnyTensor,
      expected_type: tf.DType,
      lengths: tuple[Any, ...],
      expected_values: list[Any],
  ):
    self.assertIsInstance(x, tf.RaggedTensor)
    self.assertEqual(x.dtype, expected_type)
    self.assertEqual(tf.shape(x).static_lengths(), lengths)
    x_values = np.concatenate([x_row.numpy() for x_row in x.values], axis=None)
    if x_values.dtype.kind == 'f':  # floating point
      x_values = [float('{:.2f}'.format(i)) for i in x_values]
    self.assertAllEqual(
        x_values,
        expected_values,
    )

  def test_input_pipeline_parses_features(self):
    # Collect all TF Examples in one batch. Order is random.
    input_fn = input_pipeline.make_from_disk_input_fn(
        self.record_filename,
        self.config,
        VOCABULARY_FILE,
        batch_size=3,
        shuffle=False,
    )
    input_context = tf.distribute.InputContext()
    dataset = input_fn(input_context)
    context_features, sequence_features = next(iter(dataset))

    # Check context dense features.
    self.assertRaggedValues(
        context_features['hr/cost_center'],
        tf.int64,
        [3, (2, 2, 2)],
        # [b'ORD', b'SYD', b'ORD, b'NYC, b'CDG', b'SYD']
        [3, 2, 3, 0, 1, 2],
    )

    # Check context sequence features.
    # The second example does not have code features.
    self.assertRaggedValues(
        context_features['code/reviewer_f/t'],
        tf.int64,
        [3, (2, 0, 3)],
        # [b'reviewer1', b'reviewer2', b'reviewer1', b'reviewer3', b'reviewer8']
        [1, 3, 1, 4, 2],
    )
    self.assertRaggedValues(
        context_features['code/reviewer_f/w'],
        tf.float32,
        [3, (2, 0, 3)],
        [1, 1, 1, 2, 2],
    )

    # Check ragged sequence features.
    # Example 2 is context only so sequence_features are empty.
    self.assertRaggedValues(
        sequence_features['network/prin/t'],
        tf.int64,
        [3, (1, 0, 3), (2, 1, 3, 0)],
        # [b'up4', b'up5', b'up1', b'up2', b'up3', b'up5']
        [1, 3, 4, 5, 2, 3],
    )
    self.assertRaggedValues(
        sequence_features['network/prin/w'],
        tf.float32,
        [3, (1, 0, 3), (2, 1, 3, 0)],
        [1, 0, 1, 2, 3, 1],
    )

    # Check dense features.
    self.assertRaggedValues(
        sequence_features['bash/cmd'],
        tf.float32,
        [3, (1, 0, 2), (1, 1, 1), 2],
        [1, 2, 3, 4, 5, 6],
    )

    self.assertDenseValues(
        context_features['calendar/mtg'], tf.float32, [1, 2, 1, 2, 3, 4]
    )

    # Check principal hashing is within bucket size.
    self.assertAllLessEqual(
        context_features['p'], input_pipeline._HASH_BUCKET_SIZE
    )


if __name__ == '__main__':
  tf.test.main()
