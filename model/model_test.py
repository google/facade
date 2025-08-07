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

import numpy as np
import tensorflow as tf

from model import model
from model.common import configuration
from model.common import test_utils
from model.input import pipeline
from model.optimization import optimizer
from protos import config_pb2


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


class ModelTest(tf.test.TestCase):

  def assertEmbeddingDimEquality(self, x: np.array, y: np.array):
    x_embed_dim = x.shape[-1]
    y_embed_dim = y.shape[-1]
    self.assertGreater(x_embed_dim, 0)
    self.assertEqual(x_embed_dim, y_embed_dim)

  def test_makes_trains_predicts_saves_reloads_evaluates_serves(self):
    model_config = configuration.read_config('model/test_model_config.textproto')
    facade = model.FacadeModel(model_config, VOCABULARY_FILE)

    record_filename = test_utils.write_example_file('model/input/test_examples.textproto')
    dataset_fn = pipeline.make_from_disk_input_fn(
        tf_record_pattern=record_filename,
        model_config=model_config,
        vocabulary_filepattern=VOCABULARY_FILE,
        batch_size=model_config.training_hyperparameters.batch_size,
    )
    input_context = tf.distribute.InputContext()
    dataset = dataset_fn(input_context)

    # Training.
    facade.compile(
        optimizer=optimizer.make_optimizer(
            model_config.training_hyperparameters
        ),
        steps_per_execution=2,
    )
    facade.fit(dataset, epochs=3, steps_per_epoch=10)

    # Inference with Model.call().
    x = next(iter(dataset))
    y = facade(x)
    self.assertNotEmpty(y)

    # Saving and exporting with a default serving signature.
    savedir = self.create_tempdir().full_path
    # This exports the 'serving_default' signature, see
    # https://www.tensorflow.org/api_docs/python/tf/saved_model/save#args
    facade.save(savedir, signatures=facade.predict_from_serialized_examples)

    # Reload and perform same inference as above.
    facade_reloaded = tf.keras.models.load_model(savedir)
    y2 = facade_reloaded(x)
    self.assertAllClose(y, y2)

    # Evaluation.
    facade_eval = model.FacadeModel(model_config, VOCABULARY_FILE)
    facade_eval.add_evaluation_layers()
    facade_eval.compile(steps_per_execution=2)
    facade_eval.load_weights(savedir)
    facade_eval.evaluate(dataset, steps=10)

    # Serving signature works.
    serving_fn = facade_reloaded.signatures['serving_default']
    reader = tf.data.TFRecordDataset([record_filename])
    serialized_examples = list(serialized for serialized in reader)
    y3 = serving_fn(inputs=serialized_examples)
    self.assertNotEmpty(y3)
    # y3's context must have three embedding vectors.
    context_embeddings = y3['context_embeddings'].numpy()
    self.assertLen(context_embeddings, 3)
    drive_scores = y3['drive_scores'].numpy()
    linhelm_scores = y3['bash_scores'].numpy()
    uberproxy_scores = y3['network_scores'].numpy()
    # All actions must have positive scores and correct numbers.
    self.assertAllGreater(drive_scores, 0.0)
    self.assertLen(drive_scores, 2)
    self.assertAllGreater(linhelm_scores, 0.0)
    self.assertLen(linhelm_scores, 3)
    self.assertAllGreater(uberproxy_scores, 0.0)
    self.assertLen(uberproxy_scores, 4)
    drive_embeddings = y3['drive_action_embeddings'].numpy()
    linhelm_embeddings = y3['bash_action_embeddings'].numpy()
    uberproxy_embeddings = y3['network_action_embeddings'].numpy()
    # All actions must have the same number of embeddings as scores.
    self.assertLen(drive_embeddings, 2)
    self.assertLen(linhelm_embeddings, 3)
    self.assertLen(uberproxy_embeddings, 4)
    # All action and context embeddings must have the same size.
    self.assertEmbeddingDimEquality(drive_embeddings, linhelm_embeddings)
    self.assertEmbeddingDimEquality(drive_embeddings, uberproxy_embeddings)
    self.assertEmbeddingDimEquality(drive_embeddings, context_embeddings)

  def test_serializes(self):
    model_config = configuration.read_config('model/test_model_config.textproto')
    facade = model.FacadeModel(model_config, VOCABULARY_FILE)
    model.FacadeModel.from_config(facade.get_config())


if __name__ == '__main__':
  tf.test.main()
