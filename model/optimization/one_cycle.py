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
"""Implements the One Cycle Learning Rate Schedule."""

import enum
import math
from typing import Any

import tensorflow as tf
from tensorflow import keras


# Also inherits from `int` to enable Keras layer serialization.
class Interpolation(int, enum.Enum):
  COSINE = 1
  LINEAR = 2


def _linear_interpolate(y0: float, y1: float, x: tf.Tensor) -> tf.Tensor:
  return y0 + (y1 - y0) * x


def _cos_interpolate(y0: float, y1: float, x: tf.Tensor) -> tf.Tensor:
  return y1 + (y0 - y1) / 2 * (tf.math.cos(math.pi * x) + 1.0)


@keras.saving.register_keras_serializable('decay')
class OneCycle(keras.optimizers.schedules.LearningRateSchedule):
  """Implements the One Cycle schedule in https://arxiv.org/abs/1708.07120.

  More precisely, this implements a variant with only two
  phases while the original paper has 3 phases. FastAI claims this variant
  outperforms the original version in practice. See more discussions:
  https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html,
  https://fastai1.fast.ai/callbacks.one_cycle.html#Training-with-the-1cycle-policy.

  The 2-phase schedule ramps up then down the learning rate and uses either a
  cosine or linear-shaped schedule.
  """

  def __init__(
      self,
      total_training_steps: int,
      peak_learning_rate: float,
      learning_rate_rampup_factor: float,
      learning_rate_rampdown_factor: float,
      rampup: float,
      interpolation: Interpolation,
  ):
    """Initializes the schedule.

    Args:
      total_training_steps: The total number of training steps.
      peak_learning_rate: The maximal learning rate.
      learning_rate_rampup_factor: Determines the initial learning rate as
        `peak_learning_rate / learning_rate_rampup_factor`.
      learning_rate_rampdown_factor: Determines the final learning rate as
        `peak_learning_rate / learning_rate_rampdown_factor`.
      rampup: If in [0; 1), interpreted as the proportion of total steps spent
        in the increase phase. If greater or equal to 1, interpreted as the
        absolute number of steps spent in the learning rate increase phase, in
        which case should not be greater than total_training_steps.
      interpolation: The shape of the learning rate between start and peak, and
        peak and end.

    Raises:
      ValueError: At invalid argument values.
    """
    super().__init__()
    if total_training_steps <= 0:
      raise ValueError(
          f'total_training_steps must be positive. Got {total_training_steps}.'
      )
    if learning_rate_rampup_factor <= 0:
      raise ValueError(
          'learning_rate_rampup_factor must be positive. Got'
          f' {learning_rate_rampup_factor}.'
      )
    if learning_rate_rampdown_factor <= 0:
      raise ValueError(
          'learning_rate_rampdown_factor must be positive. Got'
          f' {learning_rate_rampdown_factor}.'
      )
    if not 0 <= rampup <= total_training_steps:
      raise ValueError(
          'rampup must be non-negative and not exceed'
          f' total_training_steps. Got {rampup}.'
      )

    self.total_training_steps = total_training_steps
    self.peak_learning_rate = peak_learning_rate
    self.learning_rate_rampup_factor = learning_rate_rampup_factor
    self.learning_rate_rampdown_factor = learning_rate_rampdown_factor
    self.rampup = rampup
    self.interpolation = interpolation

    self.initial_learning_rate = (
        peak_learning_rate / learning_rate_rampup_factor
    )
    self.final_learning_rate = (
        peak_learning_rate / learning_rate_rampdown_factor
    )
    if rampup < 1:
      self.peak = total_training_steps * rampup
    else:
      self.peak = rampup
    self.end = float(total_training_steps)
    match self.interpolation:
      case Interpolation.COSINE:
        self.interpolate = _cos_interpolate
      case Interpolation.LINEAR:
        self.interpolate = _linear_interpolate
      case _:
        raise ValueError(
            f'Interpolation not implemented for: {self.interpolation}'
        )

  def get_config(self) -> dict[str, Any]:
    return {
        'total_training_steps': self.total_training_steps,
        'peak_learning_rate': self.peak_learning_rate,
        'learning_rate_rampup_factor': self.learning_rate_rampup_factor,
        'learning_rate_rampdown_factor': self.learning_rate_rampdown_factor,
        'rampup': self.rampup,
        'interpolation': self.interpolation,
    }

  def __call__(self, step: tf.Tensor) -> tf.Tensor:
    step = tf.cast(step, tf.float32)
    # Training job may slightly exceed desired number of training steps.
    step = tf.minimum(step, self.end)
    learning_rate = tf.where(
        step < self.peak,
        self.interpolate(
            self.initial_learning_rate,
            self.peak_learning_rate,
            step / self.peak,
        ),
        self.interpolate(
            self.peak_learning_rate,
            self.final_learning_rate,
            (step - self.peak) / (self.end - self.peak),
        ),
    )
    return learning_rate
