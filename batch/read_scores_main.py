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
"""Helper to read scores from the TFRecord output."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import tensorflow as tf

from protos import score_pb2


SCORE_FILE = flags.DEFINE_string(
    'score_file', '', 'TFRecord file of Score protos.'
)
TOP_N = flags.DEFINE_integer(
    'top_n', 20, 'Prints the top N scores.'
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  reader = tf.data.TFRecordDataset([SCORE_FILE.value])
  scores = []
  for raw_record in reader:
    score = score_pb2.Score()
    score.ParseFromString(raw_record.numpy())
    scores.append(score)
  scores.sort(key=lambda x: x.score, reverse=True)
  for i in range(TOP_N.value):
    print(scores[i])
  

if __name__ == '__main__':
  app.run(main)
