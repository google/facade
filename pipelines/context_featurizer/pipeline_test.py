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

import datetime
import unittest

from common import time_utils
from google.protobuf import text_format
from pipelines.context_featurizer.pipeline import build_bipartite_features
from pipelines.context_featurizer.pipeline_test_utils import assert_proto2_equal_ignoring_fields
from pipelines.context_featurizer.pipeline_test_utils import make_bipartite_graph_config
from pipelines.context_featurizer.pipeline_test_utils import make_context
from pipelines.context_featurizer.pipeline_test_utils import make_context_source_config
from pipelines.context_featurizer.pipeline_test_utils import make_peer_attribute
from pipelines.context_featurizer.pipeline_test_utils import make_peer_feature_config
from protos import context_pb2
from protos import context_source_config_pb2
from protos import contextualized_actions_pb2


class BuildBipartiteGraphTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.duration = datetime.timedelta(hours=120)

    self.t_1 = time_utils.time_ftz(2023, 7, 15, 22, 0, 0)
    self.t0 = time_utils.time_ftz(2023, 8, 1, 10, 0, 0)
    self.t1 = time_utils.time_ftz(2023, 8, 1, 22, 0, 0)
    self.t2 = time_utils.time_ftz(2023, 8, 2, 10, 0, 0)
    self.t3 = time_utils.time_ftz(2023, 8, 3, 20, 0, 0)
    self.t4 = time_utils.time_ftz(2023, 8, 4, 10, 0, 0)
    self.t5 = time_utils.time_ftz(2023, 8, 4, 22, 0, 0)
    self.t6 = time_utils.time_ftz(2023, 8, 5, 10, 0, 0)

    self.st0 = time_utils.time_ftz(2023, 8, 3, 10, 0, 0)
    self.st1 = self.t6

    self.cl1_u3_t_1 = make_peer_attribute(
        "cl", "1", 3.0, self.t_1, context_pb2.PeerAttribute.Direction.D_BACKWARD
    )
    self.cl0_u0_t0 = make_peer_attribute(
        "cl", "0", 1.0, self.t0, context_pb2.PeerAttribute.Direction.D_FORWARD
    )
    self.cl0_u1_t1 = make_peer_attribute(
        "cl", "0", 2.0, self.t1, context_pb2.PeerAttribute.Direction.D_BACKWARD
    )
    self.cl0_u1_t2 = make_peer_attribute(
        "cl", "0", 3.0, self.t2, context_pb2.PeerAttribute.Direction.D_BACKWARD
    )
    self.cl0_u2_t3 = make_peer_attribute(
        "cl", "0", 4.0, self.t3, context_pb2.PeerAttribute.Direction.D_BACKWARD
    )
    self.cl0_u0_t3 = make_peer_attribute(
        "cl", "0", 5.0, self.t3, context_pb2.PeerAttribute.Direction.D_FORWARD
    )
    self.cl0_u1_t3 = make_peer_attribute(
        "cl", "0", 6.0, self.t3, context_pb2.PeerAttribute.Direction.D_BACKWARD
    )
    self.cal0_u0_t3 = make_peer_attribute(
        "cal", "0", 7.0, self.t3, context_pb2.PeerAttribute.Direction.D_UNSET
    )
    self.cal0_u1_t4 = make_peer_attribute(
        "cal", "0", 8.0, self.t4, context_pb2.PeerAttribute.Direction.D_UNSET
    )
    self.cal0_u3_t4 = make_peer_attribute(
        "cal", "0", 9.0, self.t4, context_pb2.PeerAttribute.Direction.D_UNSET
    )
    self.cl1_u2_t4 = make_peer_attribute(
        "cl", "1", 10.0, self.t4, context_pb2.PeerAttribute.Direction.D_FORWARD
    )
    self.cl1_u0_t5 = make_peer_attribute(
        "cl", "1", 20.0, self.t5, context_pb2.PeerAttribute.Direction.D_BACKWARD
    )
    self.cl1_u3_t6 = make_peer_attribute(
        "cl", "1", 30.0, self.t6, context_pb2.PeerAttribute.Direction.D_BACKWARD
    )

    self.u3_t_1 = make_context(
        "u3", self.t_1, peer_attributes=[self.cl1_u3_t_1]
    )
    self.u0_t0 = make_context("u0", self.t0, peer_attributes=[self.cl0_u0_t0])
    self.u1_t1 = make_context("u1", self.t1, peer_attributes=[self.cl0_u1_t1])
    self.u1_t2 = make_context("u1", self.t2, peer_attributes=[self.cl0_u1_t2])
    self.u2_t3 = make_context("u2", self.t3, peer_attributes=[self.cl0_u2_t3])
    self.u0_t3 = make_context(
        "u0", self.t3, peer_attributes=[self.cl0_u0_t3, self.cal0_u0_t3]
    )
    self.u1_t3 = make_context("u1", self.t3, peer_attributes=[self.cl0_u1_t3])
    self.u1_t4 = make_context("u1", self.t4, peer_attributes=[self.cal0_u1_t4])
    self.u3_t4 = make_context("u3", self.t4, peer_attributes=[self.cal0_u3_t4])
    self.u2_t4 = make_context("u2", self.t4, peer_attributes=[self.cl1_u2_t4])
    self.u0_t5 = make_context("u0", self.t5, peer_attributes=[self.cl1_u0_t5])
    self.u3_t6 = make_context("u3", self.t6, peer_attributes=[self.cl1_u3_t6])

    self.u_config = make_bipartite_graph_config(
        [context_source_config_pb2.BipartiteGraph.TraversalMode.TM_UNDIRECTED],
        self.duration,
        context_source_config_pb2.BipartiteGraph.EdgeWeightingMethod.EWM_LATEST,
    )
    self.fb_config = make_bipartite_graph_config(
        [
            context_source_config_pb2.BipartiteGraph.TraversalMode.TM_FORWARD,
            context_source_config_pb2.BipartiteGraph.TraversalMode.TM_BACKWARD,
        ],
        self.duration,
        context_source_config_pb2.BipartiteGraph.EdgeWeightingMethod.EWM_LATEST,
    )
    self.agg_method = (
        context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod.AGG_ACCUMULATE
    )

  def test_empty_in_empty_out(self):
    pfc_cl = make_peer_feature_config("cl", 10, self.fb_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc_cl])
    contexts = []
    res = build_bipartite_features(contexts, config, [self.st0, self.st1])
    self.assertEqual(res, [])

  def test_works(self):
    pfc_cl = make_peer_feature_config("cl", 10, self.fb_config, self.agg_method)
    pfc_cal = make_peer_feature_config(
        "cal", 10, self.u_config, self.agg_method
    )
    config = make_context_source_config(self.duration, [pfc_cl, pfc_cal])

    contexts = [
        self.u3_t_1,
        self.u0_t0,
        self.u1_t1,
        self.u1_t2,
        self.u2_t3,
        self.u0_t3,
        self.u1_t3,
        self.u1_t4,
        self.u3_t4,
        self.u2_t4,
        self.u0_t5,
        self.u3_t6,
    ]
    res = build_bipartite_features(contexts, config, [self.st0, self.st1])

    # Expected results
    u0_st0 = text_format.Parse(
        """
        valid_from { seconds: 1691056800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_b"
            bag_of_weighted_words { tokens { token: "u1" } }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )
    u1_st0 = text_format.Parse(
        """
        valid_from { seconds: 1691056800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_f"
            bag_of_weighted_words { tokens { token: "u0" } }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )
    u0_st1 = text_format.Parse(
        """
        valid_from { seconds: 1691229600 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_f"
            bag_of_weighted_words { tokens { token: "u2" } }
          }
          features {
            name: "cl_b"
            bag_of_weighted_words {
              tokens { token: "u1" }
              tokens { token: "u2" }
            }
          }
          features {
            name: "cal_u"
            bag_of_weighted_words {
              tokens { token: "u0" }
              tokens { token: "u1" }
              tokens { token: "u3" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )
    u1_st1 = text_format.Parse(
        """
        valid_from { seconds: 1691229600 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_f"
            bag_of_weighted_words { tokens { token: "u0" } }
          }
          features {
            name: "cal_u"
            bag_of_weighted_words {
              tokens { token: "u0" }
              tokens { token: "u1" }
              tokens { token: "u3" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )
    u2_st1 = text_format.Parse(
        """
        valid_from { seconds: 1691229600 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_f"
            bag_of_weighted_words { tokens { token: "u0" } }
          }
          features {
            name: "cl_b"
            bag_of_weighted_words { tokens { token: "u0" } }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )
    u3_st1 = text_format.Parse(
        """
        valid_from { seconds: 1691229600 }
        features_per_source {
          source_type: "test"
          features {
            name: "cal_u"
            bag_of_weighted_words {
              tokens { token: "u0" }
              tokens { token: "u1" }
              tokens { token: "u3" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    expected_results = [
        ("u0", u0_st0),
        ("u1", u1_st0),
        ("u0", u0_st1),
        ("u1", u1_st1),
        ("u2", u2_st1),
        ("u3", u3_st1),
    ]

    self.assertEqual(len(res), len(expected_results))


    expected_results = {
        "u0": [u0_st0, u0_st1],
        "u1": [u1_st0, u1_st1],
        "u2": [u2_st1],
        "u3": [u3_st1],
    }

    # Group actual results by user ID
    grouped_res = {}
    for user_id, proto in res:
      if user_id not in grouped_res:
        grouped_res[user_id] = []
      grouped_res[user_id].append(proto)

    self.assertEqual(len(grouped_res), len(expected_results))

    ignored_fields = [
        "features_per_source.features.bag_of_weighted_words.tokens.weight"
    ]
    ignore_order_in_fields = [
        "features_per_source.features",
        "features_per_source.features.bag_of_weighted_words.tokens",
    ]

    for user_id in expected_results:
      self.assertIn(user_id, grouped_res)
      self.assertEqual(len(grouped_res[user_id]), len(expected_results[user_id]))
      for i in range(len(expected_results[user_id])):
        assert_proto2_equal_ignoring_fields(
            expected_results[user_id][i],
            grouped_res[user_id][i],
            ignored_fields,
            ignore_order_in_fields,
        )


if __name__ == "__main__":
  unittest.main()

