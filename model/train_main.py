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
"""Trains or evaluates a Facade model.

Training:
python -m model.train_main \
  --train_file=sample/train.tfrecord \
  --vocabulary_file=sample/vocabs.tfrecord \
  --model_config=sample/config.textproto \
  --model_dir=sample/model

Evaluation:
python -m model.train_main \
  --eval_file=sample/train.tfrecord \
  --vocabulary_file=sample/vocabs.tfrecord \
  --model_config=sample/config.textproto \
  --model_dir=sample/model \
  --is_evaluation_task
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from model import model as model_lib
from model.common import configuration
from model.input import pipeline
from model.optimization import optimizer
from protos import config_pb2


TRAIN_FILE = flags.DEFINE_string(
    'train_file', '', 'Input file of training data. Expected to be a TFRecord '
    'file of SequenceExamples.'
)
EVAL_FILE = flags.DEFINE_string(
    'eval_file', '', 'Input file of evaluation data. Expected to be a TFRecord '
    'file of SequenceExamples.'
)
VOCABULARY_FILE = flags.DEFINE_string(
    'vocabulary_file', '', 'Input file of vocabulary data. Expected to be a '
    'TFRecord file of Vocab protos.'
)
MODEL_CONFIG_FILE = flags.DEFINE_string(
    'model_config', '', 'Path to model configuration textproto.'
)
MODEL_DIR = flags.DEFINE_string('model_dir', '', 'Model output directory.')
IS_EVALUATION_TASK = flags.DEFINE_bool(
    'is_evaluation_task', False, 'Whether to run as the evaluation task.'
)


def get_dataset(
    model_config: config_pb2.ModelHyperparameters,
    strategy: tf.distribute.Strategy,
) -> tf.data.Dataset:
  """Constructs the dataset for training or evaluation."""
  for_evaluation = IS_EVALUATION_TASK.value
  filename = (
      EVAL_FILE.value
      if for_evaluation
      else TRAIN_FILE.value
  )
  batch_size = (
      model_config.training_hyperparameters.evaluation.batch_size
      if for_evaluation
      else model_config.training_hyperparameters.batch_size
  )
  training = not for_evaluation
  dataset_fn = pipeline.make_from_disk_input_fn(
      tf_record_pattern=filename,
      model_config=model_config,
      vocabulary_filepattern=VOCABULARY_FILE.value,
      batch_size=batch_size,
      training=training,
  )
  return strategy.distribute_datasets_from_function(dataset_fn)


class SaveLatestModels(tf.keras.callbacks.Callback):
  """Saves full model every N epochs for full recovery.

  In case the model training is interruped or restoring from the middle of the
  training is desired.
  """

  def __init__(self, every_n_epochs: int = 5):
    super().__init__()
    self.model_dir = MODEL_DIR.value
    self.every_n_epochs = every_n_epochs

  def on_epoch_end(self, epoch: int, logs=None) -> None:
    if (epoch + 1) % self.every_n_epochs == 0:
      logging.info('Saving model at epoch %s', epoch + 1)
      model_path = os.path.join(
          self.model_dir, f'intermediate_saved_model/epoch_{epoch + 1}/'
      )
      self.model.save(model_path)


def train(
    model_config: config_pb2.ModelHyperparameters,
    strategy: tf.distribute.Strategy,
    dataset: tf.data.Dataset,
):
  """Runs a training task."""
  epochs = (
      model_config.training_hyperparameters.training_examples
      // model_config.training_hyperparameters.evaluation.examples_per_training_epoch
  )
  logging.info('Number of training epochs: %s', epochs)

  steps_per_epoch = (
      model_config.training_hyperparameters.evaluation.examples_per_training_epoch
      // model_config.training_hyperparameters.batch_size
  )
  logging.info('Number of steps per epoch: %s', steps_per_epoch)

  save_path = os.path.join(MODEL_DIR.value, 'export/final/')
  save_callback = SaveLatestModels()

  with strategy.scope():
    model = model_lib.FacadeModel(model_config, VOCABULARY_FILE.value)
    model.compile(
        optimizer=optimizer.make_optimizer(
            model_config.training_hyperparameters
        ),
        # Amortizes the expensive lookup of the concrete train_function by
        # executing multiple consecutive training steps at every call.
        steps_per_execution=32,
    )
    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[save_callback],
    )
    logging.info('Saving final model to: %s', save_path)
    # This exports the 'serving_default' signature, see
    # https://www.tensorflow.org/api_docs/python/tf/saved_model/save#args
    model.save(save_path, signatures=model.predict_from_serialized_examples)
    logging.info('Model summary:\n %s', model.summary())


def evaluate(
    model_config: config_pb2.ModelHyperparameters,
    dataset: tf.data.Dataset,
):
  """Runs a sidecar evaluator task."""
  model = model_lib.FacadeModel(model_config, VOCABULARY_FILE.value)
  model.add_evaluation_layers()
  model.compile(
      optimizer=optimizer.make_optimizer(model_config.training_hyperparameters),
      steps_per_execution=32,
  )
  model.load_weights(os.path.join(MODEL_DIR.value, 'export/final'))
  steps_per_epoch = (
      model_config.training_hyperparameters.evaluation.examples_per_evaluation_epoch
      // model_config.training_hyperparameters.evaluation.batch_size
  )
  model.evaluate(dataset, steps=steps_per_epoch)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_config = configuration.read_config(MODEL_CONFIG_FILE.value)
  logging.info('Using model configuration: %s', model_config)

  strategy = tf.distribute.get_strategy()
  logging.info(
      'Using `tf.distribute.Strategy` class: `%s`',
      strategy.__class__.__name__,
  )

  dataset = get_dataset(
      model_config, strategy
  )
  if IS_EVALUATION_TASK.value:
    evaluate(model_config, dataset)
  else:
    train(model_config, strategy, dataset)


if __name__ == '__main__':
  app.run(main)
