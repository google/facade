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

from common import tf_example
from protos import contextualized_actions_pb2


class ToTFInputTest(unittest.TestCase):

  def test_context_features(self):
    ca_text = r"""
      principal: "principal_id"
      context {
        features_per_source {
          source_type: "context_type2"
          features {
            name: "bow_feature"
            bag_of_weighted_words {
              tokens { token: "a" weight: 1.0 }
              tokens { token: "b" weight: 2.0 }
            }
          }
        }
      }
    """
    contextualized_actions = text_format.Parse(
        ca_text, contextualized_actions_pb2.ContextualizedActions())

    result = tf_example.to_tf_input(contextualized_actions)

    expected_text = r"""
      context {
      feature {
        key: "p"
        value { bytes_list { value: [ "principal_id" ] } }
      }
      feature {
        key: "context_type2/bow_feature/t"
        value { bytes_list { value: [ "a", "b" ] } }
      }
      feature {
        key: "context_type2/bow_feature/w"
        value { float_list { value: [ 1.0, 2.0 ] } }
      }
    }
    """
    expected = text_format.Parse(expected_text, tf.train.SequenceExample())

    self.assertEqual(result, expected)


  def test_action_features(self):
    ca_text = r"""
      principal: "principal_id"
      actions {
        source_type: "type1"
        actions {
          features {
            name: "segment1"
            bag_of_weighted_words {
              tokens { token: "a" weight: 1.0 }
              tokens { token: "b" weight: 2.0 }
            }
          }
        }
        actions {
          features {
            name: "segment1"
            bag_of_weighted_words { tokens { token: "c" weight: 16.0 } }
          }
        }
      }
    """
    contextualized_actions = text_format.Parse(
        ca_text, contextualized_actions_pb2.ContextualizedActions())

    result = tf_example.to_tf_input(contextualized_actions)

    expected_text = r"""
      context {
        feature {
          key: "p"
          value { bytes_list { value: [ "principal_id" ] } }
        }
      }
      feature_lists {
        feature_list {
          key: "type1/segment1/t"
          value {
            feature { bytes_list { value: [ "a", "b" ] } }
            feature { bytes_list { value: [ "c" ] } }
          }
        }
        feature_list {
          key: "type1/segment1/w"
          value {
            feature { float_list { value: [ 1.0, 2.0 ] } }
            feature { float_list { value: [ 16.0 ] } }
          }
        }
      }
    """
    expected = text_format.Parse(expected_text, tf.train.SequenceExample())

    self.assertEqual(result, expected)


  def test_respects_empty_features(self):
    ca_text = r"""
      principal: "principal_id"
      context {
        features_per_source {
          source_type: "context_type2"
          features {
            name: "bow_feature"
            bag_of_weighted_words {}
          }
        }
      }
      actions {
        source_type: "type1"
        actions {
          features {
            name: "segment1"
            bag_of_weighted_words {}
          }
        }
        actions {
          features {
            name: "segment1"
            bag_of_weighted_words {}
          }
        }
      }
    """
    contextualized_actions = text_format.Parse(
        ca_text, contextualized_actions_pb2.ContextualizedActions())

    result = tf_example.to_tf_input(contextualized_actions)

    expected_text = r"""
      context {
        feature {
          key: "p"
          value { bytes_list { value: [ "principal_id" ] } }
        }
        feature {
          key: "context_type2/bow_feature/t"
          value { bytes_list {} }
        }
        feature {
          key: "context_type2/bow_feature/w"
          value { float_list {} }
        }
      }
      feature_lists {
        feature_list {
          key: "type1/segment1/t"
          value {
            feature { bytes_list {} }
            feature { bytes_list {} }
          }
        }
        feature_list {
          key: "type1/segment1/w"
          value {
            feature { float_list {} }
            feature { float_list {} }
          }
        }
      }
    """
    expected = text_format.Parse(expected_text, tf.train.SequenceExample())

    self.assertEqual(result, expected)

    
if __name__ == '__main__':
    unittest.main()
