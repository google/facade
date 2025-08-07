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
"""Gets Facade scores from an already trained model.

python -m batch.inference_main \
  --directive=sample/directive.textproto \
  --start_time="2024-07-08 00:00:00" \
  --end_time="2024-07-15 00:00:00" \
  --action_path=sample/action.tfrecord \
  --context_path=sample/context.tfrecord \
  --output_file=sample/scores.tfrecord \
  --model_config=sample/config.textproto \
  --model_dir=sample/model
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow import keras

from batch import batch_lib
from common import directive_utils
from common import tf_example
from common import time_utils
from common.pipeline_type import PipelineType
from model import model as model_lib
from model.common import configuration
from model.input import pipeline
from model.optimization import one_cycle
from protos import config_pb2
from protos import score_pb2


DIRECTIVE_FILE = flags.DEFINE_string(
    'directive', '', 'Path to directive configuration textproto.'
)
START_TIME = flags.DEFINE_string(
    'start_time', '2025-01-01 00:00:00',
    'Generate the dataset from contexts and actions whose time '
    'falls at or after this value. Formatted as YYYY-MM-DD HH:MM:SS in UTC.'
)
END_TIME = flags.DEFINE_string(
    'end_time', '2025-01-02 00:00:00',
    'Generate the dataset from contexts and actions whose time '
    'falls before this value. Formatted as YYYY-MM-DD HH:MM:SS in UTC.'
)
ACTION_FILE_PATH = flags.DEFINE_string(
    'action_path', '', 'File path to a TFRecord file of facade Action protos.'
)
CONTEXT_FILE_PATH = flags.DEFINE_string(
    'context_path', '', 'File path to a TFRecord file of facade Context protos.'
)
OUTPUT_FILE = flags.DEFINE_string(
    'output_file', '', 'File path to write scores. Output is a TFRecord file of '
    'Score protos.'
)
MODEL_CONFIG_FILE = flags.DEFINE_string(
    'model_config', '', 'Path to model configuration textproto.'
)
MODEL_DIR = flags.DEFINE_string('model_dir', '', 'Model output directory.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if DIRECTIVE_FILE.value == '':
    raise app.UsageError('Directive file must be specified')
  if ACTION_FILE_PATH.value == '':
    raise app.UsageError('Action file must be specified')
  if CONTEXT_FILE_PATH.value == '':
    raise app.UsageError('Context file must be specified')
  start_time = time_utils.parse_datetime_flag(START_TIME.value)
  end_time = time_utils.parse_datetime_flag(END_TIME.value)
  if end_time <= start_time:
    raise app.UsageError('Start time must be before end time')

  directive = directive_utils.read_directive(DIRECTIVE_FILE.value)
  logging.info('Using directive: %s', directive)
  model_config = configuration.read_config(MODEL_CONFIG_FILE.value)
  logging.info('Using model configuration: %s', model_config)

  # Create featurized actions.
  contextualized_actions = batch_lib.compute_contextualized_actions(
    directive, PipelineType.INFERENCE, start_time, end_time, ACTION_FILE_PATH.value,
    CONTEXT_FILE_PATH.value)
  sequence_examples = []
  for ca in contextualized_actions:
    sequence_examples.append(tf_example.to_tf_input(ca).SerializeToString())

  # Load model and compute scores per action source.
  with keras.utils.custom_object_scope({'OneCycle': one_cycle}):
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR.value, 'export/final'))
  serving_fn = model.signatures['serving_default']
  results = serving_fn(inputs=sequence_examples)

  # Allocate score protos for each action in the contextualized actions.
  scores = []
  for action_type in model_config.action_name_to_architecture:
    for ca in contextualized_actions:
      for ca_actions_by_source in ca.actions:
        if ca_actions_by_source.source_type != action_type:
          continue
        for a in ca_actions_by_source.actions:
          scores.append(score_pb2.Score(
            principal=ca.principal,
            action_type=action_type,
            resource_id=a.resource_id,
            action_id=a.id
          ))
  
  # Read in scores from results.
  i = 0
  for action_type in model_config.action_name_to_architecture:
    action_type_scores = results[action_type + '_scores']
    for score in action_type_scores:
      scores[i].score = score
      i += 1

  # Write scores to output file.
  with tf.io.TFRecordWriter(OUTPUT_FILE.value) as writer:
    for s in scores:
      writer.write(s.SerializeToString())


if __name__ == '__main__':
  app.run(main)
