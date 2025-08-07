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

import collections
import os
import unittest
from common import time_utils
from context import context_source
from google.protobuf import text_format
from pipelines.context_featurizer.pipeline_test_utils import assert_proto2_equal_ignoring_fields
from pipelines.context_featurizer.pipeline_test_utils import make_context
from pipelines.context_featurizer.pipeline_test_utils import make_peer_attribute
from protos import context_pb2
from protos import context_source_config_pb2
from protos import contextualized_actions_pb2
import tensorflow as tf


def make_tf_feature(value):
  """Creates a TensorFlow Feature proto from a string, float, or int."""
  ret = tf.train.Feature()
  if isinstance(value, float):
    ret.float_list.value.append(value)
  elif isinstance(value, int):
    ret.int64_list.value.append(value)
  elif isinstance(value, str):
    ret.bytes_list.value.append(value.encode("utf-8"))
  else:
    raise TypeError("Unsupported type for make_tf_feature")
  return ret


def write_tfrecord(filename, records):
  """Writes a list of Context protos to a TFRecord file."""
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  with tf.io.TFRecordWriter(filename) as writer:
    for record in records:
      writer.write(record.SerializeToString())


class ContextSourceTest(unittest.TestCase):

  def test_bipartite_only_works(self):
    temp_dir = "/tmp"
    source_filename = os.path.join(temp_dir, "bipartite.tfrecord")
    config = text_format.Parse(
        """
          type: "test"
          context_lookback { seconds: 262800 }
          peer_feature_configs {
            name: "cal"
            max_peers: 10
            aggregation_method: AGG_ACCUMULATE
            bipartite_graph {
              traversal_modes: TM_UNDIRECTED
              half_life { seconds: 262800 }
              edge_weighting_method: EWM_LATEST
            }
          }""",
        context_source_config_pb2.ContextSourceConfig(),
    )

    t0 = time_utils.time_ftz(2023, 8, 1, 10, 0, 0)
    t1 = time_utils.time_ftz(2023, 8, 2, 10, 0, 0)
    t2 = time_utils.time_ftz(2023, 8, 3, 10, 0, 0)
    t3 = time_utils.time_ftz(2023, 8, 4, 10, 0, 0)

    # Snapshot times.
    st0 = t1
    st1 = t3
    snapshot_times = [st0, st1]

    cal0_u0_t0 = make_peer_attribute(
        "cal", "0", 7.0, t0, context_pb2.PeerAttribute.Direction.D_UNSET
    )
    cal0_u1_t1 = make_peer_attribute(
        "cal", "0", 8.0, t1, context_pb2.PeerAttribute.Direction.D_UNSET
    )
    cal0_u2_t2 = make_peer_attribute(
        "cal", "0", 9.0, t2, context_pb2.PeerAttribute.Direction.D_UNSET
    )

    # At st0 = t1, only u0 is connected to u0.
    # At st1 = t3, everybody is connected by everybody.
    u0_t0 = make_context("u0", t0, [cal0_u0_t0])
    u1_t1 = make_context("u1", t1, [cal0_u1_t1])
    u2_t2 = make_context("u2", t2, [cal0_u2_t2])

    write_tfrecord(source_filename, [u0_t0, u1_t1, u2_t2])

    res = context_source.get_featurized_context(
        config, snapshot_times, source_filename
    )

    u0_st0 = text_format.Parse(
        """
          features_per_source {
            source_type: "test"
            features {
              name: "cal_u"
              bag_of_weighted_words { tokens { token: "u0" } }
            }
          }""",
        contextualized_actions_pb2.FeaturizedContext(),
    )
    u0_st1 = text_format.Parse(
        """
          features_per_source {
            source_type: "test"
            features {
              name: "cal_u"
              bag_of_weighted_words {
                tokens { token: "u0" }
                tokens { token: "u1" }
                tokens { token: "u2" }
              }
            }
          }""",
        contextualized_actions_pb2.FeaturizedContext(),
    )
    u0_st0.valid_from.CopyFrom(time_utils.convert_to_proto_time(st0))
    u0_st1.valid_from.CopyFrom(time_utils.convert_to_proto_time(st1))
    u1_st1 = u0_st1
    u2_st1 = u0_st1

    ignored_fields = [
        "features_per_source.features.bag_of_weighted_words.tokens.weight"
    ]

    ignore_order_in_fields = [
        "features_per_source.features.bag_of_weighted_words.tokens",
        "features_per_source.features",
    ]

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    self.assertEqual(len(res_dict["u0"]), 2)
    assert_proto2_equal_ignoring_fields(
        res_dict["u0"][0], u0_st0, ignored_fields, ignore_order_in_fields
    )
    assert_proto2_equal_ignoring_fields(
        res_dict["u0"][1], u0_st1, ignored_fields, ignore_order_in_fields
    )
    self.assertEqual(len(res_dict["u1"]), 1)
    assert_proto2_equal_ignoring_fields(
        res_dict["u1"][0], u1_st1, ignored_fields, ignore_order_in_fields
    )
    self.assertEqual(len(res_dict["u2"]), 1)
    assert_proto2_equal_ignoring_fields(
        res_dict["u2"][0], u2_st1, ignored_fields, ignore_order_in_fields
    )


if __name__ == "__main__":
  unittest.main()

