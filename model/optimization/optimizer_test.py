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
import unittest
from collections.abc import Sequence

from google.protobuf import text_format
from model.optimization import optimizer
from protos import config_pb2
from parameterized import parameterized


def make_test_cases() -> Sequence[config_pb2.ModelTrainingHyperparameters]:
  learning_rate_schedules = [
      """exponential_decay {
          initial_learning_rate: 1.0
          decay_steps: 10
          decay_rate: 0.5
          staircase: True
          }""",
      """one_cycle {
          peak_learning_rate: 1.0
          learning_rate_rampup_factor: 10.0
          learning_rate_rampdown_factor: 100.0
          rampup: 0.5
          interpolation: I_COSINE
          }""",
      """one_cycle {
          peak_learning_rate: 1.0
          learning_rate_rampup_factor: 10.0
          learning_rate_rampdown_factor: 100.0
          rampup: 20
          interpolation: I_LINEAR
          }""",
  ]
  optimizers = [
      """sgd {}""",
      """sgd {
          momentum: 0.01
          global_clipnorm: 100.0
          }""",
      """adam_w {}""",
      """adam_w {
          weight_decay: 0.001
          beta_1: 0.95
          beta_2: 0.9999
          epsilon: 0.0001
          global_clipnorm: 100.0
          }""",
  ]
  result = []
  for lrs in learning_rate_schedules:
    for opt in optimizers:
      textproto = (
          "batch_size: 1\ntraining_examples: 100\nlearning_rate_schedule"
          f" \n{{{lrs}\n}}\noptimizer\n{{{opt}\n}}"
      )
      config = config_pb2.ModelTrainingHyperparameters()
      text_format.Parse(textproto, config)
      result.append(config)
  return result


class OptimizerTest(unittest.TestCase):

  @parameterized.expand(make_test_cases())
  def test_makes_onecycle(self, config):
    opt = optimizer.make_optimizer(config)
    self.assertIsNotNone(opt)


if __name__ == "__main__":
  unittest.main()
