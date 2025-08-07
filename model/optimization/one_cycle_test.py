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
import inspect
import math

import tensorflow as tf

from model.optimization import one_cycle as oc
from parameterized import parameterized


def linear_interpolate(start, end, loc):
  return start + (end - start) * loc


def cos_interpolate(start, end, loc):
  return end + (start - end) * (math.cos(math.pi * loc) + 1) / 2


def reference(
    step: int,
    total_training_steps: int,
    peak_learning_rate: float,
    learning_rate_rampup_factor: float,
    learning_rate_rampdown_factor: float,
    rampup: float,
    interpolation: oc.Interpolation,
) -> float:
  match interpolation:
    case oc.Interpolation.COSINE:
      interpolate = cos_interpolate
    case oc.Interpolation.LINEAR:
      interpolate = linear_interpolate
    case _:
      raise ValueError('Internal Error.')

  if step >= total_training_steps:
    return peak_learning_rate / learning_rate_rampdown_factor
  if step <= 0:
    return peak_learning_rate / learning_rate_rampup_factor
  peak_step = total_training_steps * rampup if rampup < 1 else rampup
  if step < peak_step:
    start = peak_learning_rate / learning_rate_rampup_factor
    end = peak_learning_rate
    loc = step / peak_step
  else:
    start = peak_learning_rate
    end = peak_learning_rate / learning_rate_rampdown_factor
    loc = (step - peak_step) / (total_training_steps - peak_step)
  return interpolate(start, end, loc)


class OneCycleTest(tf.test.TestCase):

  def test_serializes(self):
    config = {
        'total_training_steps': 1234,
        'peak_learning_rate': 0.5,
        'learning_rate_rampup_factor': 10,
        'learning_rate_rampdown_factor': 100,
        'rampup': 0.25,
        'interpolation': oc.Interpolation.LINEAR,
    }
    f = oc.OneCycle.from_config(config)
    self.assertEqual(f.get_config(), config)
    self.assertCountEqual(
        inspect.signature(oc.OneCycle).parameters.keys(),
        config.keys(),
    )

  @parameterized.expand([
    (10, 0.25, oc.Interpolation.LINEAR),
    (79, 0.25, oc.Interpolation.LINEAR),
    (10, 6.0, oc.Interpolation.LINEAR),
    (79, 6.0, oc.Interpolation.LINEAR),
    (10, 0.25, oc.Interpolation.COSINE),
    (79, 0.25, oc.Interpolation.COSINE),
    (10, 6.0, oc.Interpolation.COSINE),
    (79, 6.0, oc.Interpolation.COSINE),
  ])
  def test_works(self, total_training_steps, rampup, interpolation):
    peak_learning_rate = 0.5
    learning_rate_rampup_factor = 10
    learning_rate_rampdown_factor = 100
    f = oc.OneCycle(
        total_training_steps,
        peak_learning_rate,
        learning_rate_rampup_factor,
        learning_rate_rampdown_factor,
        rampup,
        interpolation,
    )
    # Test past the last training step.
    for step in range(total_training_steps + 10):
      self.assertAllClose(
          f(tf.constant(step, dtype=tf.int32)),
          reference(
              step,
              total_training_steps,
              peak_learning_rate,
              learning_rate_rampup_factor,
              learning_rate_rampdown_factor,
              rampup,
              interpolation,
          ),
      )


if __name__ == '__main__':
  tf.test.main()
