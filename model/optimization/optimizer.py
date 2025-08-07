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
"""Defines the model optimizer."""

import tensorflow as tf

from model.optimization import one_cycle
from model.optimization import sparse_sgd
from protos import config_pb2


def make_optimizer(
    training_config: config_pb2.ModelTrainingHyperparameters,
) -> tf.keras.optimizers.Optimizer:
  """Constructs the optimizer as specified by the configuration."""
  match training_config.learning_rate_schedule.WhichOneof('schedule'):
    case 'exponential_decay':
      config = training_config.learning_rate_schedule.exponential_decay
      schedule = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=config.initial_learning_rate,
          decay_steps=config.decay_steps,
          decay_rate=config.decay_rate,
          staircase=config.staircase,
      )
    case 'one_cycle':
      config = training_config.learning_rate_schedule.one_cycle
      match config.interpolation:
        case config_pb2.LearningRateSchedule.OneCycle.Interpolation.I_COSINE:
          interpolation = one_cycle.Interpolation.COSINE
        case config_pb2.LearningRateSchedule.OneCycle.Interpolation.I_LINEAR:
          interpolation = one_cycle.Interpolation.LINEAR
        case _:
          raise ValueError(
              f'Unrecognized interpolation: {config.interpolation}.'
          )

      train_steps = (
          training_config.training_examples // training_config.batch_size
      )
      schedule = one_cycle.OneCycle(
          total_training_steps=train_steps,
          peak_learning_rate=config.peak_learning_rate,
          learning_rate_rampup_factor=config.learning_rate_rampup_factor,
          learning_rate_rampdown_factor=config.learning_rate_rampdown_factor,
          rampup=config.rampup,
          interpolation=interpolation,
      )
    case _:
      raise ValueError(
          'Unrecognized learning rate schedule:'
          f' {training_config.learning_rate_schedule}.'
      )

  match training_config.optimizer.WhichOneof('optimizer'):
    case 'sgd':
      config = training_config.optimizer.sgd
      clip = config.global_clipnorm if config.global_clipnorm > 0 else None
      weight_decay = (
          training_config.weight_decay
          if training_config.weight_decay >= 0
          else 0.0
      )
      momentum = config.momentum if config.momentum > 0 else 0.0
      return sparse_sgd.SGDSp(
          learning_rate=schedule,
          sparse_weight_decay=weight_decay,
          sparse_momentum=momentum,
          global_clipnorm=clip,
      )

    case 'adam_w':
      config = training_config.optimizer.adam_w
      clip = config.global_clipnorm if config.global_clipnorm > 0 else None
      weight_decay = (
          training_config.weight_decay
          if training_config.weight_decay >= 0
          else 0.004
      )
      return tf.keras.optimizers.AdamW(
          learning_rate=schedule,
          weight_decay=weight_decay,
          beta_1=config.beta_1 if config.beta_1 > 0 else 0.9,
          beta_2=config.beta_2 if config.beta_2 > 0 else 0.999,
          epsilon=config.epsilon if config.epsilon > 0 else 1e-7,
          global_clipnorm=clip,
      )

    case _:
      raise ValueError(f'Unrecognized optimizer: {training_config.optimizer}.')
