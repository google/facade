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
import tensorflow as tf
from google.protobuf import text_format
from typing import List

from common import vocab
from protos import vocab_pb2


class VocabTest(unittest.TestCase):

  def create_vocab(self, name: str, is_context: bool, values: List[str]) -> vocab_pb2.Vocab:
    return vocab_pb2.Vocab(
        name = name,
        is_context = is_context,
        values = [v.encode() for v in values]
    )

  def test_works(self):
    input1_text = r"""
      context {
        feature {
          key: "hr/cost_center"
          value { bytes_list { value: "cc1" } }
        }
        feature {
          key: "hr/manager"
          value { bytes_list { value: "m1" value: "m2" } }
        }
      }
      feature_lists {
        feature_list {
          key: "drive/parameter"
          value { feature { bytes_list { value: "v1" } } }
        }
        feature_list {
          key: "drive/parameter.agg_segment_size"
          value { feature { int64_list { value: 1 } } }
        }
      }
    """
    input1 = text_format.Parse(input1_text, tf.train.SequenceExample())

    result = vocab.extract_vocabs([input1])

    v1 = self.create_vocab("hr/cost_center", True, ["cc1"]);
    v2 = self.create_vocab("hr/manager", True, ["m2", "m1"]);
    v3 = self.create_vocab("drive/parameter", False, ["v1"]);

    self.assertEqual(result, [v1, v2, v3])


  def test_merges(self):
    input1_text = r"""
      context {
        feature {
          key: "hr/cost_center"
          value { bytes_list { value: "cc1" } }
        }
        feature {
          key: "hr/manager"
          value { bytes_list { value: "m1" value: "m2" } }
        }
      }
      feature_lists {
        feature_list {
          key: "drive/parameter"
          value { feature { bytes_list { value: "v1" } } }
        }
        feature_list {
          key: "drive/parameter.agg_segment_size"
          value { feature { int64_list { value: 1 } } }
        }
      }
    """
    input1 = text_format.Parse(input1_text, tf.train.SequenceExample())
    input2_text = r"""
      context {
        feature {
          key: "hr/cost_center"
          value { bytes_list { value: "cc2" } }
        }
        feature {
          key: "hr/manager"
          value { bytes_list { value: "m3" value: "m2" } }
        }
      }
      feature_lists {
        feature_list {
          key: "drive/parameter"
          value { feature { bytes_list { value: "v1" } } }
        }
        feature_list {
          key: "drive/parameter.agg_segment_size"
          value { feature { int64_list { value: 1 } } }
        }
        feature_list {
          key: "uberproxy/parameter"
          value { feature { bytes_list { value: "v2" } } }
        }
      }
    """
    input2 = text_format.Parse(input2_text, tf.train.SequenceExample())

    result = vocab.extract_vocabs([input1, input2])

    v1 = self.create_vocab("hr/cost_center", True, ["cc1", "cc2"]);
    v2 = self.create_vocab("hr/manager", True, ["m3", "m2", "m1"]);
    v3 = self.create_vocab("drive/parameter", False, ["v1"]);
    v4 = self.create_vocab("uberproxy/parameter", False, ["v2"]);

    self.assertEqual(result, [v1, v2, v3, v4])
    

if __name__ == '__main__':
  unittest.main()

