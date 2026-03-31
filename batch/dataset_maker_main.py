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
"""Generates training and validation data, or runs the model to get scores.

Training data:
python -m batch.dataset_maker_main \
  --directive=sample/directive.textproto \
  --start_time="2024-04-01 00:00:00" \
  --end_time="2024-07-01 00:00:00" \
  --action_path=sample/action.tfrecord \
  --context_path=sample/context.tfrecord \
  --train_output=sample/train.tfrecord

Validation data:
python -m batch.dataset_maker_main \
  --directive=sample/directive.textproto \
  --start_time="2024-07-01 00:00:00" \
  --end_time="2024-07-08 00:00:00" \
  --action_path=sample/action.tfrecord \
  --context_path=sample/context.tfrecord \
  --validation_output=sample/validation.tfrecord
"""

from absl import app
from absl import flags
from absl import logging
from collections.abc import Sequence
import datetime
import os
import random
import time
import tensorflow as tf

from batch import batch_lib
from common import directive_utils
from common.pipeline_type import PipelineType
from common import string_pool
from common import tf_example
from common import time_utils
from common import vocab


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
TRAIN_OUTPUT = flags.DEFINE_string(
    'train_output', '', 'Output path where training tf.SequenceExamples are '
    'written. If present, a vocab file is also created with _vocab appended to the path.'
)
VALIDATION_OUTPUT = flags.DEFINE_string(
    'validation_output', '', 'Output path where validation tf.SequenceExamples are written.'
)


def _format_elapsed(seconds: float) -> str:
  """Format elapsed time in a human-readable way."""
  if seconds < 60:
    return f'{seconds:.0f}s'
  elif seconds < 3600:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f'{mins}m {secs}s'
  else:
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f'{hours}h {mins}m'


def write_tf_record_file(filename, data):
  with tf.io.TFRecordWriter(filename) as writer:
    for d in data:
      writer.write(d.SerializeToString())


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  overall_start_time = time.time()

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
  
  # Calculate time range
  time_range_days = (end_time - start_time).days
  
  train_output_path = TRAIN_OUTPUT.value
  validation_output_path = VALIDATION_OUTPUT.value
  if train_output_path and validation_output_path:
    raise app.UsageError('Only one of train_output or validation_output should be specified')

  pipeline_type = PipelineType.TRAINING
  if validation_output_path:
    pipeline_type = PipelineType.VALIDATION

  # Log header
  logging.info('* Starting Dataset Generation')
  logging.info('Time range: %s to %s (%d days)', start_time, end_time, time_range_days)
  logging.info('Pipeline type: %s', pipeline_type.name)
  logging.info('Action sources: %d', len(directive.action_sources))
  logging.info('Context sources: %d', len(directive.context_sources))
  
  logging.info('')
  logging.info('* Stage 1: Computing Contextualized Actions')
  stage_start_time = time.time()

  contextualized_actions = batch_lib.compute_contextualized_actions(
    directive, pipeline_type, start_time, end_time, ACTION_FILE_PATH.value,
    CONTEXT_FILE_PATH.value)
  
  stage_elapsed = time.time() - stage_start_time
  logging.info('Completed contextualization in %s', _format_elapsed(stage_elapsed))
  
  # Log string pool statistics
  pool_stats = string_pool.get_pool_stats()
  logging.info('String pool - Principals: %s, Resources: %s, Action IDs: %s',
               f"{pool_stats['principals']:,}",
               f"{pool_stats['resources']:,}",
               f"{pool_stats['action_ids']:,}")
  
  logging.info('')
  logging.info('* Stage 2: Converting to TF SequenceExamples')
  stage_start_time = time.time()
  
  sequence_examples = []
  for ca in contextualized_actions:
    sequence_examples.append(tf_example.to_tf_input(ca))

  stage_elapsed = time.time() - stage_start_time
  logging.info('Created %s sequence examples in %s', 
               f'{len(sequence_examples):,}', _format_elapsed(stage_elapsed))
  
  logging.info('')
  logging.info('* Stage 3: Shuffling Examples')
  stage_start_time = time.time()

  # Always shuffles tf.SequenceExample before writing.
  random.shuffle(sequence_examples)
  
  stage_elapsed = time.time() - stage_start_time
  logging.info('Shuffled %s examples in %s', 
               f'{len(sequence_examples):,}', _format_elapsed(stage_elapsed))

  # VALIDATION
  if pipeline_type == PipelineType.VALIDATION:
    logging.info('')
    logging.info('* Stage 4: Writing Validation Data')
    stage_start_time = time.time()
    
    write_tf_record_file(validation_output_path, sequence_examples)
    
    stage_elapsed = time.time() - stage_start_time
    file_size_mb = os.path.getsize(validation_output_path) / 1024**2
    logging.info('Wrote %s records to %s (%.1f MB) in %s',
                 f'{len(sequence_examples):,}',
                 validation_output_path,
                 file_size_mb,
                 _format_elapsed(stage_elapsed))
    
    overall_elapsed = time.time() - overall_start_time
    logging.info('')
    logging.info('* Dataset Generation Complete')
    logging.info('Total time: %s', _format_elapsed(overall_elapsed))
    logging.info('Output: %s (%s records)', validation_output_path, f'{len(sequence_examples):,}')
    return

  # TRAINING
  logging.info('')
  logging.info('* Stage 4: Extracting Vocabulary')
  stage_start_time = time.time()
  
  vocabs = vocab.extract_vocabs(sequence_examples)
  
  stage_elapsed = time.time() - stage_start_time
  logging.info('Extracted vocabulary in %s', _format_elapsed(stage_elapsed))
  
  logging.info('')
  logging.info('* Stage 5: Writing Training Data')
  stage_start_time = time.time()
  
  vocab_path = os.path.join(os.path.dirname(train_output_path), 'vocabs.tfrecord')
  write_tf_record_file(vocab_path, vocabs)
  write_tf_record_file(train_output_path, sequence_examples)
  
  stage_elapsed = time.time() - stage_start_time
  train_size_mb = os.path.getsize(train_output_path) / 1024**2
  vocab_size_mb = os.path.getsize(vocab_path) / 1024**2
  logging.info('Wrote %s training records to %s (%.1f MB) in %s',
               f'{len(sequence_examples):,}',
               train_output_path,
               train_size_mb,
               _format_elapsed(stage_elapsed))
  logging.info('Wrote vocabulary to %s (%.1f MB)', vocab_path, vocab_size_mb)
  
  overall_elapsed = time.time() - overall_start_time
  logging.info('')
  logging.info('* Dataset Generation Complete')
  logging.info('Total time: %s', _format_elapsed(overall_elapsed))
  logging.info('Output: %s (%s records)', train_output_path, f'{len(sequence_examples):,}')
  logging.info('Vocabulary: %s', vocab_path)


if __name__ == '__main__':
  app.run(main)
