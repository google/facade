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
"""This module defines the data input pipeline."""

from typing import Callable, Mapping, MutableMapping

from absl import logging
import tensorflow as tf

from model.architectures import token_embeddings as te
from model.common import write_once_sorted_dict as wosd
from protos import config_pb2


AnyFeatureSpec = (
    tf.io.FixedLenFeature
    | tf.io.VarLenFeature
    | tf.io.RaggedFeature
    | tf.io.FixedLenSequenceFeature
)
AnyTensor = tf.Tensor | tf.SparseTensor | tf.RaggedTensor
AnyMapping = Mapping[str, AnyTensor]
_HASH_BUCKET_SIZE = 2_000_000_000


def _segment_reduction_parsing_spec(
    segment_reduction: config_pb2.SegmentReduction,
) -> wosd.WriteOnceSortedDict[str, AnyFeatureSpec]:
  """Generates the parsing spec for a segment reduction."""
  spec = wosd.WriteOnceSortedDict()
  # Emit token parsing spec.
  if not segment_reduction.token_feature_name:
    raise ValueError(
        'Segment reductions must have a token feature name:'
        f' {segment_reduction}'
    )
  spec[segment_reduction.token_feature_name] = tf.io.RaggedFeature(tf.string)
  # Emit weight parsing spec, if one is present.
  if segment_reduction.intensity_feature_name:
    if not segment_reduction.intensity_feature_name:
      raise ValueError(
          'Segment reductions has intesities but no intesity feature name:'
          f' {segment_reduction}'
      )
    spec[segment_reduction.intensity_feature_name] = tf.io.RaggedFeature(
        tf.float32
    )
  return spec


def _fixed_size_dense_feature_parsing_spec(
    dense_feature: config_pb2.FixedSizeDenseFeature, is_sequence: bool
) -> wosd.WriteOnceSortedDict[str, AnyFeatureSpec]:
  """Generates the parsing spec for a fixed size dense feature."""
  spec = wosd.WriteOnceSortedDict()
  if not dense_feature.feature_name:
    raise ValueError(
        f'Dense features must have a feature name: {dense_feature}'
    )
  if is_sequence:
    # Use UniformRowLength to force the last dimension of the tensor to be
    # fixed to the dense feature size. This introduces a size 1 dimension above
    # it that a later pipeline will have to remove.
    spec[dense_feature.feature_name] = tf.io.RaggedFeature(
        tf.float32,
        partitions=[tf.io.RaggedFeature.UniformRowLength(dense_feature.size)],
    )
  else:
    spec[dense_feature.feature_name] = tf.io.FixedLenFeature(
        dense_feature.size, tf.float32
    )
  return spec


def _concatenate_then_snn_parsing_spec(
    architecture: config_pb2.Architecture.ConcatenateThenSNN, is_sequence: bool
) -> Mapping[str, AnyFeatureSpec]:
  """Creates parsing spec for a ConcatenateThenSNN architecture."""
  spec = wosd.WriteOnceSortedDict()
  for segment_reduction in architecture.segment_reductions:
    spec.update(_segment_reduction_parsing_spec(segment_reduction))
  for dense_feature in architecture.fixed_size_dense_features:
    spec.update(
        _fixed_size_dense_feature_parsing_spec(dense_feature, is_sequence)
    )
  return spec


def make_parsing_spec(
    model_config: config_pb2.ModelHyperparameters,
) -> tuple[
    MutableMapping[str, AnyFeatureSpec], MutableMapping[str, AnyFeatureSpec]
]:
  """Creates the parsing spec from a model configuration.

  Args:
    model_config: Model configuration.

  Returns:
    spec: Parsing spec containing the tf Feature specifications.
  """
  context_spec = wosd.WriteOnceSortedDict()
  sequence_spec = wosd.WriteOnceSortedDict()

  # All examples must contain a principal feature indicating the principal that
  # is contextualized.
  if not model_config.principal_feature_name:
    raise ValueError('Principal feature name is required')
  context_spec[model_config.principal_feature_name] = tf.io.FixedLenFeature(
      1, tf.string
  )

  match model_config.context_architecture.WhichOneof('architecture'):
    case 'concatenate_then_snn':
      context_spec.update(
          _concatenate_then_snn_parsing_spec(
              model_config.context_architecture.concatenate_then_snn,
              is_sequence=False,
          )
      )
    case _:
      raise ValueError(
          f'Unknown architecture {model_config.context_architecture}.'
      )

  for architecture in model_config.action_name_to_architecture.values():
    match architecture.WhichOneof('architecture'):
      case 'concatenate_then_snn':
        sequence_spec.update(
            _concatenate_then_snn_parsing_spec(
                architecture.concatenate_then_snn,
                is_sequence=True,
            )
        )
      case _:
        raise ValueError(f'Unknown architecture: {architecture}.')

  return dict(context_spec), dict(sequence_spec)


def _apply_principal_hashing(
    context_inputs: AnyMapping,
    principal_feature_name: str,
    bucket_size: int = _HASH_BUCKET_SIZE,
) -> AnyMapping:
  """Hash principal strings to int32 values."""
  context_inputs = dict(context_inputs)
  principals = context_inputs[principal_feature_name]
  check_has_principals = tf.debugging.assert_greater(
      tf.strings.length(principals), 0
  )
  with tf.control_dependencies([check_has_principals]):
    # A random salt is added to the principal so that the deterministic hashing
    # generates different values for the same principal at each batch.
    salt = tf.random.uniform([], minval=0, maxval=2_000_000_000)
    salt = tf.strings.as_string(salt)
    salted_principals = tf.strings.join([principals, salt])
    principal_hash = tf.strings.to_hash_bucket_fast(
        salted_principals, bucket_size
    )
    context_inputs[principal_feature_name] = tf.cast(principal_hash, tf.int32)
  return context_inputs


def make_from_disk_input_fn(
    tf_record_pattern: str,
    model_config: config_pb2.ModelHyperparameters,
    vocabulary_filepattern: str,
    batch_size: int,
    training: bool = False,
    filename_shuffle_buffer_size: int = 10000,
    parallel_open_files: int = 25,
    examples_per_file_factor: int = 10,
    example_shuffle_buffer_size_factor: int = 25,
    shuffle: bool = True,
) -> Callable[[tf.distribute.InputContext], tf.data.Dataset]:
  """Creates function that can be used to initialize a TrainSpec or EvalSpec.

  Besides reading, parsing and integerizing string tokens, the function is also
  responsible for good randomization of examples through mixing of examples from
  different files and shuffling examples in limited size buffers.

  Args:
    tf_record_pattern: File pattern for (sharded) TFRecords of TF Examples.
    model_config: Model configuration from which the specifications of inputs
      that are expected in the TF Examples is derived.
    vocabulary_filepattern: This pattern matches the token vocabulary files.
      Vocabulary files contain the set of all distinct tokens per TF Feature
      name and are used to perform the string to integer lookup pre-processing.
    batch_size: Number of examples per batch.
    training: If true, apply oov token conversion to string lookup by oov_rate
      for each embedding name.
    filename_shuffle_buffer_size: Controls quality of shuffling. Larger buffers
      are better but use more memory.
    parallel_open_files: Controls quality of shuffling. Larger values are better
      but use more memory.
    examples_per_file_factor: Controls quality of shuffling. Larger values are
      better but use more memory.
    example_shuffle_buffer_size_factor: Controls quality of shuffling. Larger
      values are better but use more memory.
    shuffle: Whether to shuffle the records before batching. Should only be
      False to ensure deterministic behavior in tests.

  Returns:
    input_fn: A function that takes no arguments and returns a TF Dataset,
      consisting of a dict of tensors.

  Raises:
    IOError: If tf_record_pattern does not match any file.
  """
  if not tf.io.gfile.glob(tf_record_pattern):
    raise IOError(f'No matching file for pattern {tf_record_pattern}.')

  context_spec, sequence_spec = make_parsing_spec(model_config)
  print_lines = ['Using context parsing spec:']
  for k, v in context_spec.items():
    print_lines.append(f'\t {k}: {v}')
  print_lines.append('Using sequence parsing spec:')
  for k, v in sequence_spec.items():
    print_lines.append(f'\t {k}: {v}')
  logging.info('\n'.join(print_lines))

  token_lookup_embedders = te.TokenLookupEmbedders(
      vocabulary_filepattern, model_config
  )

  def input_fn(input_context: tf.distribute.InputContext) -> tf.data.Dataset:
    """Creates ops to read, parse and shuffle serialized tf.Examples from disk.

    Args:
      input_context: specifies properties of the global input pipeline, such as
        the total number of workers reading the data.

    Returns:
      dataset: A tf dataset of dict of tensors.
    """
    with tf.name_scope('disk_read_parse'):
      # Randomly shuffled tf.Dataset of matching filenames.
      filenames = tf.data.Dataset.list_files(tf_record_pattern)
      # Files are deterministically assigned to workers.
      filenames = filenames.shard(
          input_context.num_input_pipelines, input_context.input_pipeline_id
      )
      filenames = filenames.shuffle(filename_shuffle_buffer_size)
      records = filenames.interleave(
          tf.data.TFRecordDataset,
          cycle_length=parallel_open_files,
          block_length=batch_size * examples_per_file_factor,
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False,
      )

      if shuffle:
        records = records.shuffle(
            batch_size * example_shuffle_buffer_size_factor
        )
      records = records.repeat()

      # For distributed parameter server training, the batch size is independent
      # of the number of workers.
      records = records.batch(
          batch_size,
          drop_remainder=True,
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False,
      )
      examples = records.map(
          lambda batch: tf.io.parse_sequence_example(  # pylint: disable=g-long-lambda
              batch, context_spec, sequence_spec
          ),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False,
      )
      examples = examples.map(
          lambda context, sequence, _: token_lookup_embedders.apply_string_lookup(
              context, sequence, training
          ),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False,
      )
      examples = examples.map(
          lambda context, sequence: (
              _apply_principal_hashing(
                  context, model_config.principal_feature_name
              ),
              sequence,
          ),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False,
      )
      examples = examples.prefetch(tf.data.AUTOTUNE)

    return examples

  return input_fn
