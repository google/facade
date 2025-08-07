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
import datetime
import math
import unittest

from common import time_utils
from google.protobuf import text_format
from pipelines.context_featurizer import pipeline_utils
from pipelines.context_featurizer.pipeline_test_utils import assert_proto2_equal_ignoring_fields
from pipelines.context_featurizer.pipeline_test_utils import make_bipartite_graph_config
from pipelines.context_featurizer.pipeline_test_utils import make_context
from pipelines.context_featurizer.pipeline_test_utils import make_context_source_config
from pipelines.context_featurizer.pipeline_test_utils import make_peer_attribute
from pipelines.context_featurizer.pipeline_test_utils import make_peer_feature_config
from protos import context_pb2
from protos import context_source_config_pb2
from protos import contextualized_actions_pb2


class ComputeDiscountTest(unittest.TestCase):

  def test_works_on_past(self):
    # 24h difference and 6h half life: 1/16
    attribute_time = time_utils.time_ftz(2021, 1, 20, 12, 0, 0)
    snapshot_time = time_utils.time_ftz(2021, 1, 21, 12, 0, 0)
    half_life = datetime.timedelta(hours=6)
    self.assertAlmostEqual(
        pipeline_utils.compute_discount(
            attribute_time, snapshot_time, half_life
        ),
        0.0625,
    )

  def test_returns_one_on_zero_duration(self):
    attribute_time = time_utils.time_ftz(2021, 1, 20, 12, 0, 0)
    snapshot_time = time_utils.time_ftz(2021, 1, 21, 12, 0, 0)
    half_life = datetime.timedelta(0)
    self.assertAlmostEqual(
        pipeline_utils.compute_discount(
            attribute_time, snapshot_time, half_life
        ),
        1.0,
    )

  def test_works_on_future(self):
    # 12h difference and 6h half life: 1/4
    attribute_time = time_utils.time_ftz(2021, 1, 21, 0, 0, 0)
    snapshot_time = time_utils.time_ftz(2021, 1, 20, 12, 0, 0)
    half_life = datetime.timedelta(hours=6)
    self.assertAlmostEqual(
        pipeline_utils.compute_discount(
            attribute_time, snapshot_time, half_life
        ),
        0.25,
    )

  def test_dies_on_negative_duration(self):
    attribute_time = time_utils.time_ftz(2021, 1, 20, 12, 0, 0)
    snapshot_time = time_utils.time_ftz(2021, 1, 21, 12, 0, 0)
    # Makes a negative duration.
    half_life = attribute_time - snapshot_time
    with self.assertRaisesRegex(ValueError, "Negative"):
      pipeline_utils.compute_discount(attribute_time, snapshot_time, half_life)


class GetSnapshotTimesTest(unittest.TestCase):

  def test_returns_empty_when_non_overlapping(self):
    context_time_past = time_utils.time_ftz(2023, 8, 28, 0, 0, 0)
    context_time_future = time_utils.time_ftz(2023, 8, 30, 0, 0, 1)
    duration = datetime.timedelta(hours=24)
    snapshot_times = [
        time_utils.time_ftz(2023, 8, 29, 0, 0, 1),
        time_utils.time_ftz(2023, 8, 30, 0, 0, 1),
    ]
    snapshot_intervals = []
    for st in snapshot_times:
      snapshot_intervals.append((st - duration, st))

    res_past = pipeline_utils.get_snapshot_times(
        context_time_past, snapshot_intervals
    )
    self.assertEqual(len(res_past), 0)
    res_future = pipeline_utils.get_snapshot_times(
        context_time_future, snapshot_intervals
    )
    self.assertEqual(len(res_future), 0)

  def test_works(self):
    context_time = time_utils.time_ftz(2023, 8, 28, 0, 0, 0)
    duration = datetime.timedelta(hours=24)
    snapshot_times = [
        time_utils.time_ftz(2023, 8, 27, 16, 0, 0),
        time_utils.time_ftz(2023, 8, 28, 0, 0, 0),
        time_utils.time_ftz(2023, 8, 28, 8, 0, 0),
        time_utils.time_ftz(2023, 8, 28, 16, 0, 0),
        time_utils.time_ftz(2023, 8, 29, 0, 0, 0),
        time_utils.time_ftz(2023, 8, 29, 8, 0, 0),
    ]
    snapshot_intervals = []
    for st in snapshot_times:
      snapshot_intervals.append((st - duration, st))
    # All snapshot_times in (2023-08-28T00:00:00, 2023-08-29T00:00:00].
    res = pipeline_utils.get_snapshot_times(context_time, snapshot_intervals)
    expected = [
        time_utils.time_ftz(2023, 8, 28, 8, 0, 0),
        time_utils.time_ftz(2023, 8, 28, 16, 0, 0),
        time_utils.time_ftz(2023, 8, 29, 0, 0, 0),
    ]
    self.assertEqual(res, expected)


def make_context_undirected(
    context: context_pb2.Context,
) -> context_pb2.Context:
  undirected = context_pb2.Context()
  undirected.CopyFrom(context)
  if undirected.peer_attributes:
    undirected.peer_attributes[0].direction = (
        context_pb2.PeerAttribute.Direction.D_UNSET
    )
  return undirected


class ReduceBipartitePeerAttributesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Context event timeline (time flows from left to right).
    #  ----------- st0 -------------- st1 -------
    #  - t0 --- t1 --- t2 -- t3 ----- t4 --- t5 -

    # Context event happens at these times.
    self.t0 = time_utils.time_ftz(2023, 8, 1, 10, 0, 0)
    self.t1 = time_utils.time_ftz(2023, 8, 1, 22, 0, 0)
    self.t2 = time_utils.time_ftz(2023, 8, 2, 10, 0, 0)
    self.t3 = time_utils.time_ftz(2023, 8, 3, 20, 0, 0)
    self.t4 = time_utils.time_ftz(2023, 8, 4, 10, 0, 0)
    self.t5 = time_utils.time_ftz(2023, 8, 4, 22, 0, 0)

    # We construct Context at these times.
    # st0 = t1 + 6h = t2 - 6h
    self.st0 = time_utils.time_ftz(2023, 8, 2, 4, 0, 0)
    # st1 = t4 = t3 + 14h = t5 - 12h
    self.st1 = time_utils.time_ftz(2023, 8, 4, 10, 0, 0)

    # PeerAttribute convention: <name>_<value>_<principal>_<time>
    self.cl_0_u0_t0 = make_peer_attribute("cl", "0", 0.0, self.t0)
    self.cl_0_u0_t1 = make_peer_attribute("cl", "0", 2.0, self.t1)
    self.cl_0_u0_t2 = make_peer_attribute("cl", "0", 3.0, self.t2)
    self.cl_0_u0_t3 = make_peer_attribute("cl", "0", 4.0, self.t3)
    self.cl_0_u0_t4 = make_peer_attribute("cl", "0", 5.0, self.t4)
    self.cl_0_u0_t5 = make_peer_attribute("cl", "0", 6.0, self.t5)

    self.cl_0_u1_t2 = make_peer_attribute("cl", "0", 1.0, self.t2)
    self.cl_0_u1_t3 = make_peer_attribute("cl", "0", 2.0, self.t3)
    self.cl_0_u1_t4 = make_peer_attribute("cl", "0", 3.0, self.t4)

    self.cl_1_u0_t2 = make_peer_attribute("cl", "1", 10.0, self.t2)
    self.cl_1_u0_t3 = make_peer_attribute("cl", "1", 20.0, self.t3)
    self.cl_1_u0_t4 = make_peer_attribute("cl", "1", 30.0, self.t4)
    self.cl_1_u0_t5 = make_peer_attribute("cl", "1", 40.0, self.t5)

    self.cal_0_u0_t1 = make_peer_attribute("cal", "0", 0.1, self.t1)
    self.cal_0_u0_t2 = make_peer_attribute("cal", "0", 0.2, self.t2)
    self.cal_1_u0_t2 = make_peer_attribute("cal", "1", 0.3, self.t2)
    self.cal_1_u0_t3 = make_peer_attribute("cal", "1", 0.4, self.t3)

    # Context convention: <principal>_<time>
    self.u0_t0 = make_context("u0", self.t0, [self.cl_0_u0_t0])
    self.u0_t1 = make_context(
        "u0", self.t1, [self.cl_0_u0_t1, self.cal_0_u0_t1]
    )
    self.u0_t2 = make_context(
        "u0",
        self.t2,
        [self.cl_0_u0_t2, self.cl_1_u0_t2, self.cal_0_u0_t2, self.cal_1_u0_t2],
    )
    self.u0_t3 = make_context(
        "u0", self.t3, [self.cl_0_u0_t3, self.cl_1_u0_t3, self.cal_1_u0_t3]
    )
    self.u0_t4 = make_context("u0", self.t4, [self.cl_0_u0_t4, self.cl_1_u0_t4])
    self.u0_t5 = make_context("u0", self.t5, [self.cl_0_u0_t5, self.cl_1_u0_t5])

    self.u1_t2 = make_context("u1", self.t2, [self.cl_0_u1_t2])
    self.u1_t3 = make_context("u1", self.t3, [self.cl_0_u1_t3])
    self.u1_t4 = make_context("u1", self.t4, [self.cl_0_u1_t4])

    # Impurity source type.
    self.unmatched_type_context = make_context(
        "u2", self.t2, [self.cl_0_u1_t2], "unknown_type"
    )

    self.agg_method = (
        context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod.AGG_ACCUMULATE
    )

  def peer_attribute_to_tuple(self, pa):
    return (
        pa.name,
        pa.value,
        round(pa.weight, 7),
        pa.direction,
    )

  def context_to_comparable(self, context):
    peer_attributes_comparable = collections.Counter(
        self.peer_attribute_to_tuple(pa) for pa in context.peer_attributes
    )
    return (
        context.principal,
        context.valid_from.nanos,
        peer_attributes_comparable,
    )

  def assertContextsEqual(self, res, expected):
    self.assertEqual(len(res), len(expected), "Context list lengths differ")

    # Convert lists of Contexts to lists of comparable tuples
    res_comparable = sorted([self.context_to_comparable(c) for c in res])
    expected_comparable = sorted(
        [self.context_to_comparable(c) for c in expected]
    )

    # Counters handle the order-insensitivity for peer_attributes
    self.assertEqual(res_comparable, expected_comparable)

  def test_empty_on_empty(self):
    lookback_duration = datetime.timedelta(hours=1)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [], config, [self.st0]
    )
    self.assertEqual(len(res), 0)

  def test_empty_outside_window(self):
    lookback_duration = datetime.timedelta(hours=1)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    # t0_ < st0_ - lookback_duration. Excluded.
    res_early = pipeline_utils.reduce_bipartite_peer_attributes(
        [self.u0_t0], config, [self.st0, self.st1]
    )
    # t4_ = st1_. Excluded. t5_ > st1_. Excluded.
    res_late = pipeline_utils.reduce_bipartite_peer_attributes(
        [self.u0_t4, self.u0_t5], config, [self.st0, self.st1]
    )

    self.assertEqual(len(res_early), 0)
    self.assertEqual(len(res_late), 0)

  def test_empty_non_config(self):
    lookback_duration = datetime.timedelta(hours=48)
    # Attribute's name does not appear in config.
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "NonConfig", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [
            self.u0_t0,
            self.u0_t1,
            self.u0_t2,
            self.u0_t3,
            self.u0_t4,
            self.u0_t5,
        ],
        config,
        [self.st0, self.st1],
    )
    self.assertEqual(len(res), 0)

  def test_empty_non_bipartite(self):
    lookback_duration = datetime.timedelta(hours=48)
    # Non bipartite graph config.
    peer_feature_config = make_peer_feature_config("cl", 10)
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [
            self.u0_t0,
            self.u0_t1,
            self.u0_t2,
            self.u0_t3,
            self.u0_t4,
            self.u0_t5,
        ],
        config,
        [self.st0, self.st1],
    )
    self.assertEqual(len(res), 0)

  def test_empty_on_no_events_within_lookback(self):
    # Too short lookback duration to include any contexts.
    lookback_duration = datetime.timedelta(seconds=1)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [
            self.u0_t0,
            self.u0_t1,
            self.u0_t2,
            self.u0_t3,
            self.u0_t4,
            self.u0_t5,
        ],
        config,
        [self.st0, self.st1],
    )
    self.assertEqual(len(res), 0)

  def test_works_latest(self):
    lookback_duration = datetime.timedelta(hours=48)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    # t4_ = st1_. Thus, u0_t4_ is ignored in the result.
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [self.u0_t0, self.u0_t1, self.u0_t2, self.u0_t3, self.u0_t4],
        config,
        [self.st0, self.st1],
    )

    # Picks the latest events from st0_ and st1_.
    cl_0_t1 = self.cl_0_u0_t1
    cl_0_t1.ClearField("time")
    cl_0_t3 = self.cl_0_u0_t3
    cl_0_t3.ClearField("time")
    cl_1_t3 = self.cl_1_u0_t3
    cl_1_t3.ClearField("time")

    res1 = make_context("u0", self.st0, [cl_0_t1])
    res2 = make_context("u0", self.st1, [cl_0_t3])
    res3 = make_context("u0", self.st1, [cl_1_t3])
    self.assertContextsEqual(res, [res1, res2, res3])

  def test_works_discounted_latest(self):
    lookback_duration = datetime.timedelta(hours=48)
    half_life = datetime.timedelta(hours=12)
    graph_config = make_bipartite_graph_config(
        [],
        half_life,
        context_source_config_pb2.BipartiteGraph.EWM_DISCOUNTED_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    # t4_ = st1_. Thus, u0_t4_ is ignored in the result.
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [self.u0_t0, self.u0_t1, self.u0_t2, self.u0_t3, self.u0_t4],
        config,
        [self.st0, self.st1],
    )

    # Picks the latest events from st0_ and st1_, then multiplies discount.
    cl_0_t1 = self.cl_0_u0_t1
    cl_0_t1.ClearField("time")
    cl_0_t1.weight = self.cl_0_u0_t1.weight * pipeline_utils.compute_discount(
        self.t1, self.st0, half_life
    )
    cl_0_t3 = self.cl_0_u0_t3
    cl_0_t3.ClearField("time")
    cl_0_t3.weight = self.cl_0_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )
    cl_1_t3 = self.cl_1_u0_t3
    cl_1_t3.ClearField("time")
    cl_1_t3.weight = self.cl_1_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )

    res1 = make_context("u0", self.st0, [cl_0_t1])
    res2 = make_context("u0", self.st1, [cl_0_t3])
    res3 = make_context("u0", self.st1, [cl_1_t3])
    self.assertContextsEqual(res, [res1, res2, res3])

  def test_works_sum_discounted(self):
    lookback_duration = datetime.timedelta(hours=24)
    half_life = datetime.timedelta(hours=12)
    graph_config = make_bipartite_graph_config(
        [],
        half_life,
        context_source_config_pb2.BipartiteGraph.EWM_SUM_DISCOUNTED,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    # t4_ = st1_. Thus, u0_t4_ is ignored in the result.
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [self.u0_t0, self.u0_t1, self.u0_t2, self.u0_t3, self.u0_t4],
        config,
        [self.st0, self.st1],
    )

    # Merges events within lookback_duration.
    # t0_ > st0_ - 24h. Included.
    cl_0_t1 = self.cl_0_u0_t1
    cl_0_t1.ClearField("time")
    cl_0_t1.weight = self.cl_0_u0_t0.weight * pipeline_utils.compute_discount(
        self.t0, self.st0, half_life
    ) + self.cl_0_u0_t1.weight * pipeline_utils.compute_discount(
        self.t1, self.st0, half_life
    )
    # t2_ < st1_ - 24h. Excluded.
    cl_0_t3 = self.cl_0_u0_t3
    cl_0_t3.ClearField("time")
    cl_0_t3.weight = self.cl_0_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )
    cl_1_t3 = self.cl_1_u0_t3
    cl_1_t3.ClearField("time")
    cl_1_t3.weight = self.cl_1_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )

    res1 = make_context("u0", self.st0, [cl_0_t1])
    res2 = make_context("u0", self.st1, [cl_0_t3])
    res3 = make_context("u0", self.st1, [cl_1_t3])
    self.assertContextsEqual(res, [res1, res2, res3])

  def test_works_log_sum_discounted(self):
    lookback_duration = datetime.timedelta(hours=18)
    half_life = datetime.timedelta(hours=12)
    graph_config = make_bipartite_graph_config(
        [],
        half_life,
        context_source_config_pb2.BipartiteGraph.EWM_LOG_SUM_DISCOUNTED,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    # t4_ = st1_. Thus, u0_t4_ is ignored in the result.
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [self.u0_t0, self.u0_t1, self.u0_t2, self.u0_t3, self.u0_t4],
        config,
        [self.st0, self.st1],
    )

    # Merges events within lookback_duration.
    # t0_ = st0_ - 18h. Included.
    cl_0_t1 = self.cl_0_u0_t1
    cl_0_t1.ClearField("time")
    raw_weight = self.cl_0_u0_t0.weight * pipeline_utils.compute_discount(
        self.t0, self.st0, half_life
    ) + self.cl_0_u0_t1.weight * pipeline_utils.compute_discount(
        self.t1, self.st0, half_life
    )
    cl_0_t1.weight = math.log(raw_weight + 1.0)
    # t2_ < st1_ - 18h. Excluded.
    cl_0_t3 = self.cl_0_u0_t3
    cl_0_t3.ClearField("time")
    raw_weight = self.cl_0_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )
    cl_0_t3.weight = math.log(raw_weight + 1.0)
    cl_1_t3 = self.cl_1_u0_t3
    cl_1_t3.ClearField("time")
    raw_weight = self.cl_1_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )
    cl_1_t3.weight = math.log(raw_weight + 1.0)

    res1 = make_context("u0", self.st0, [cl_0_t1])
    res2 = make_context("u0", self.st1, [cl_0_t3])
    res3 = make_context("u0", self.st1, [cl_1_t3])
    self.assertContextsEqual(res, [res1, res2, res3])

  def test_works_multiple_users(self):
    lookback_duration = datetime.timedelta(hours=60)
    half_life = datetime.timedelta(hours=12)
    graph_config = make_bipartite_graph_config(
        [],
        half_life,
        context_source_config_pb2.BipartiteGraph.EWM_SUM_DISCOUNTED,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    # t4_ = st1_. Thus, u0_t4_ and u1_t4_ are ignored in the result.
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [
            self.u0_t0,
            self.u0_t1,
            self.u0_t2,
            self.u0_t3,
            self.u0_t4,
            self.u1_t2,
            self.u1_t3,
            self.u1_t4,
        ],
        config,
        [self.st0, self.st1],
    )

    # Merges events within lookback_duration.
    # Principal "u0".
    # t0_ = st0_ - 60h. Included.
    cl_0_t1 = self.cl_0_u0_t1
    cl_0_t1.ClearField("time")
    weight = self.cl_0_u0_t0.weight * pipeline_utils.compute_discount(
        self.t0, self.st0, half_life
    ) + self.cl_0_u0_t1.weight * pipeline_utils.compute_discount(
        self.t1, self.st0, half_life
    )
    cl_0_t1.weight = weight
    # t0_ < st1_ - 60h. Excluded.
    # t1_ = st1_ - 60h. Included. t2_ > st1_ - 60h. Included.
    cl_0_t3 = self.cl_0_u0_t3
    cl_0_t3.ClearField("time")
    weight = (
        self.cl_0_u0_t1.weight
        * pipeline_utils.compute_discount(self.t1, self.st1, half_life)
        + self.cl_0_u0_t2.weight
        * pipeline_utils.compute_discount(self.t2, self.st1, half_life)
        + self.cl_0_u0_t3.weight
        * pipeline_utils.compute_discount(self.t3, self.st1, half_life)
    )
    cl_0_t3.weight = weight
    cl_1_t3 = self.cl_1_u0_t3
    cl_1_t3.ClearField("time")
    weight = self.cl_1_u0_t2.weight * pipeline_utils.compute_discount(
        self.t2, self.st1, half_life
    ) + self.cl_1_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )
    cl_1_t3.weight = weight

    # Principal "u1".
    cl_0_t3_u1 = self.cl_0_u1_t3
    cl_0_t3_u1.ClearField("time")
    weight = self.cl_0_u1_t2.weight * pipeline_utils.compute_discount(
        self.t2, self.st1, half_life
    ) + self.cl_0_u1_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )
    cl_0_t3_u1.weight = weight

    res1 = make_context("u0", self.st0, [cl_0_t1])
    res2 = make_context("u0", self.st1, [cl_0_t3])
    res3 = make_context("u0", self.st1, [cl_1_t3])
    res4 = make_context("u1", self.st1, [cl_0_t3_u1])

    expected = [res1, res2, res3, res4]
    self.assertEqual(len(res), len(expected), "Context list lengths differ")
    res_sorted = sorted(res, key=lambda x: (x.principal, x.valid_from.seconds))
    expected_sorted = sorted(
        expected, key=lambda x: (x.principal, x.valid_from.seconds)
    )

    for i in range(len(res_sorted)):
      assert_proto2_equal_ignoring_fields(
          res_sorted[i],
          expected_sorted[i],
          ignored_fields=[],
          ignore_order_in_fields=["peer_attributes"],
      )

  def test_works_multiple_attributes(self):
    lookback_duration = datetime.timedelta(hours=60)
    half_life = datetime.timedelta(hours=12)
    cl_graph_config = make_bipartite_graph_config(
        [],
        half_life,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    cal_graph_config = make_bipartite_graph_config(
        [],
        half_life,
        context_source_config_pb2.BipartiteGraph.EWM_DISCOUNTED_LATEST,
    )
    cl_peer_feature_config = make_peer_feature_config(
        "cl", 10, cl_graph_config, self.agg_method
    )
    cal_peer_feature_config = make_peer_feature_config(
        "cal", 10, cal_graph_config, self.agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [cl_peer_feature_config, cal_peer_feature_config]
    )
    # t4_ = st1_. Thus, u0_t4_ is ignored in the result.
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [self.u0_t0, self.u0_t1, self.u0_t2, self.u0_t3, self.u0_t4],
        config,
        [self.st0, self.st1],
    )

    # "cl" attribute. Picks the latest events from st0_ and st1_.
    cl_0_t1 = self.cl_0_u0_t1
    cl_0_t1.ClearField("time")
    cl_0_t3 = self.cl_0_u0_t3
    cl_0_t3.ClearField("time")
    cl_1_t3 = self.cl_1_u0_t3
    cl_1_t3.ClearField("time")

    # "cal" attribute. Picks the latest events and discount them.
    cal_0_t1 = self.cal_0_u0_t1
    cal_0_t1.ClearField("time")
    cal_0_t1.weight = self.cal_0_u0_t1.weight * pipeline_utils.compute_discount(
        self.t1, self.st0, half_life
    )
    cal_0_t2 = self.cal_0_u0_t2
    cal_0_t2.ClearField("time")
    cal_0_t2.weight = self.cal_0_u0_t2.weight * pipeline_utils.compute_discount(
        self.t2, self.st1, half_life
    )
    cal_1_t3 = self.cal_1_u0_t3
    cal_1_t3.ClearField("time")
    cal_1_t3.weight = self.cal_1_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )

    res1 = make_context("u0", self.st0, [cl_0_t1])
    res2 = make_context("u0", self.st1, [cl_0_t3])
    res3 = make_context("u0", self.st1, [cl_1_t3])
    res4 = make_context("u0", self.st0, [cal_0_t1])
    res5 = make_context("u0", self.st1, [cal_0_t2])
    res6 = make_context("u0", self.st1, [cal_1_t3])
    expected = [res1, res2, res3, res4, res5, res6]
    self.assertEqual(len(res), len(expected), "Context list lengths differ")
    res_sorted = sorted(res, key=lambda x: (x.principal, x.valid_from.seconds))
    expected_sorted = sorted(
        expected, key=lambda x: (x.principal, x.valid_from.seconds)
    )

    for i in range(len(res_sorted)):
      assert_proto2_equal_ignoring_fields(
          res_sorted[i],
          expected_sorted[i],
          ignored_fields=[],
          ignore_order_in_fields=["peer_attributes"],
      )

  def test_works_multiple_users_and_attributes(self):
    lookback_duration = datetime.timedelta(hours=60)
    half_life = datetime.timedelta(hours=12)
    cl_graph_config = make_bipartite_graph_config(
        [],
        half_life,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    cal_graph_config = make_bipartite_graph_config(
        [],
        half_life,
        context_source_config_pb2.BipartiteGraph.EWM_DISCOUNTED_LATEST,
    )
    cl_peer_feature_config = make_peer_feature_config(
        "cl", 10, cl_graph_config, self.agg_method
    )
    cal_peer_feature_config = make_peer_feature_config(
        "cal", 10, cal_graph_config, self.agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [cl_peer_feature_config, cal_peer_feature_config]
    )
    # t5_ > t4_ = st1_. Thus, u0_t4_, u0_t5_, and u1_t4_ are ignored.
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [
            self.u0_t0,
            self.u0_t1,
            self.u0_t2,
            self.u0_t3,
            self.u0_t4,
            self.u0_t5,
            self.u1_t2,
            self.u1_t3,
            self.u1_t4,
        ],
        config,
        [self.st0, self.st1],
    )

    # "cl" attribute. Picks the latest events from st0 and st1.
    # Principal "u0".
    cl_0_t1 = self.cl_0_u0_t1
    cl_0_t1.ClearField("time")
    cl_0_t3 = self.cl_0_u0_t3
    cl_0_t3.ClearField("time")
    cl_1_t3 = self.cl_1_u0_t3
    cl_1_t3.ClearField("time")
    # Principal "u1".
    cl_0_t3_u1 = self.cl_0_u1_t3
    cl_0_t3_u1.ClearField("time")

    # "cal" attribute. Picks the latest events and discount them.
    cal_0_t1 = self.cal_0_u0_t1
    cal_0_t1.ClearField("time")
    cal_0_t1.weight = self.cal_0_u0_t1.weight * pipeline_utils.compute_discount(
        self.t1, self.st0, half_life
    )
    cal_0_t2 = self.cal_0_u0_t2
    cal_0_t2.ClearField("time")
    cal_0_t2.weight = self.cal_0_u0_t2.weight * pipeline_utils.compute_discount(
        self.t2, self.st1, half_life
    )
    cal_1_t3 = self.cal_1_u0_t3
    cal_1_t3.ClearField("time")
    cal_1_t3.weight = self.cal_1_u0_t3.weight * pipeline_utils.compute_discount(
        self.t3, self.st1, half_life
    )

    res1 = make_context("u0", self.st0, [cl_0_t1])
    res2 = make_context("u0", self.st1, [cl_0_t3])
    res3 = make_context("u0", self.st1, [cl_1_t3])
    res4 = make_context("u1", self.st1, [cl_0_t3_u1])
    res5 = make_context("u0", self.st0, [cal_0_t1])
    res6 = make_context("u0", self.st1, [cal_0_t2])
    res7 = make_context("u0", self.st1, [cal_1_t3])

    expected = [res1, res2, res3, res4, res5, res6, res7]
    self.assertEqual(len(res), len(expected), "Context list lengths differ")
    res_sorted = sorted(res, key=lambda x: (x.principal, x.valid_from.seconds))
    expected_sorted = sorted(
        expected, key=lambda x: (x.principal, x.valid_from.seconds)
    )

    for i in range(len(res_sorted)):
      assert_proto2_equal_ignoring_fields(
          res_sorted[i],
          expected_sorted[i],
          ignored_fields=[],
          ignore_order_in_fields=["peer_attributes"],
      )

  def test_works_attribute_value_agg_latest(self):
    lookback_duration = datetime.timedelta(hours=240)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    agg_method = (
        context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod.AGG_LATEST
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )

    # At st0, cl = "0" is kept with weight 2.0 as of u0_t1.
    # At st1, cl = "1" is kept with weight 10.0 as of u0_t2.
    u0_t0 = make_context("u0", self.t0, [self.cl_0_u0_t0])
    u0_t1 = make_context("u0", self.t1, [self.cl_0_u0_t1])
    # Makes a peer feature of cl = "0" and time = t2 - 1second.
    # This is ignored by AGG_LATEST.
    cl_0_u0_t2 = self.cl_0_u0_t1
    cl_0_u0_t2.weight = 5.0
    cl_0_u0_t2.time.CopyFrom(
        time_utils.convert_to_proto_time(
            self.t2 - datetime.timedelta(seconds=1)
        )
    )
    u0_t2 = make_context("u0", self.t2, [cl_0_u0_t2, self.cl_1_u0_t2])
    res = pipeline_utils.reduce_bipartite_peer_attributes(
        [u0_t0, u0_t1, u0_t2], config, [self.st0, self.st1]
    )

    # Picks the latest attribute values from st0 and st1.
    cl_0_t1 = self.cl_0_u0_t1
    cl_0_t1.ClearField("time")
    cl_0_t2 = self.cl_1_u0_t2
    cl_0_t2.ClearField("time")

    res1 = make_context("u0", self.st0, [cl_0_t1])
    res2 = make_context("u0", self.st1, [cl_0_t2])
    expected = [res1, res2]
    self.assertEqual(len(res), len(expected), "Context list lengths differ")
    res_sorted = sorted(res, key=lambda x: (x.principal, x.valid_from.seconds))
    expected_sorted = sorted(
        expected, key=lambda x: (x.principal, x.valid_from.seconds)
    )

    for i in range(len(res_sorted)):
      assert_proto2_equal_ignoring_fields(
          res_sorted[i],
          expected_sorted[i],
          ignored_fields=[],
          ignore_order_in_fields=["peer_attributes"],
      )

  def test_dies_on_unknown_ewm(self):
    lookback_duration = datetime.timedelta(hours=48)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_UNSPECIFIED,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    with self.assertRaisesRegex(
        ValueError, "Unsupported edge weighting method"
    ):
      pipeline_utils.reduce_bipartite_peer_attributes(
          [self.u0_t0, self.u0_t1], config, [self.st0]
      )

  def test_dies_empty_snapshot_times(self):
    lookback_duration = datetime.timedelta(hours=48)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    with self.assertRaisesRegex(ValueError, "Empty snapshot_times"):
      pipeline_utils.reduce_bipartite_peer_attributes(
          [self.u0_t0, self.u0_t1], config, []
      )

  def test_dies_on_duplicated_snapshot_times(self):
    lookback_duration = datetime.timedelta(hours=48)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    with self.assertRaisesRegex(ValueError, "Duplicate snapshot_times"):
      pipeline_utils.reduce_bipartite_peer_attributes(
          [self.u0_t0, self.u0_t1], config, [self.st0, self.st0]
      )

  def test_dies_on_nonpositive_lookback(self):
    lookback_duration = datetime.timedelta(0)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    with self.assertRaisesRegex(
        ValueError, "context_lookback must be a positive duration."
    ):
      pipeline_utils.reduce_bipartite_peer_attributes(
          [self.u0_t0, self.u0_t1], config, [self.st0, self.st1]
      )

  def test_dies_on_unsorted_snapshot_times(self):
    lookback_duration = datetime.timedelta(hours=12)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    with self.assertRaisesRegex(
        ValueError, "Input snapshot_times must be sorted."
    ):
      pipeline_utils.reduce_bipartite_peer_attributes(
          [self.u0_t0, self.u0_t1], config, [self.st1, self.st0]
      )

  def test_dies_on_not_matching_types(self):
    lookback_duration = datetime.timedelta(hours=120)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, self.agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    with self.assertRaisesRegex(
        ValueError, "Context source types do not match."
    ):
      pipeline_utils.reduce_bipartite_peer_attributes(
          [self.u0_t0, self.u0_t1, self.unmatched_type_context],
          config,
          [self.st0, self.st1],
      )

  def test_dies_on_unset_aggregation_method(self):
    lookback_duration = datetime.timedelta(hours=120)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    agg_method = (
        context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod.AGG_UNSPECIFIED
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, agg_method
    )
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )
    with self.assertRaisesRegex(
        ValueError, "Aggregation method must be specified"
    ):
      pipeline_utils.reduce_bipartite_peer_attributes(
          [self.u0_t0, self.u0_t1], config, [self.st0, self.st1]
      )

  def test_dies_on_ambiguous_agg_latest(self):
    lookback_duration = datetime.timedelta(hours=240)
    graph_config = make_bipartite_graph_config(
        [],
        lookback_duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    agg_method = (
        context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod.AGG_LATEST
    )
    peer_feature_config = make_peer_feature_config(
        "cl", 10, graph_config, agg_method
    )
    # Only cl attributes get processed.
    config = make_context_source_config(
        lookback_duration, [peer_feature_config]
    )

    # At st0, cl = "0" is kept with weight 2.0 as of u0_t1.
    # At st1, cl = "0" and cl = "1" are present. Dies there.
    u0_t0 = make_context("u0", self.t0, [self.cl_0_u0_t0])
    u0_t1 = make_context("u0", self.t1, [self.cl_0_u0_t1])
    cl_0_u0_t2 = self.cl_0_u0_t1
    cl_0_u0_t2.time.CopyFrom(time_utils.convert_to_proto_time(self.t2))
    u0_t2 = make_context("u0", self.t2, [cl_0_u0_t2, self.cl_1_u0_t2])
    with self.assertRaisesRegex(ValueError, "Ambiguous AGG_LATEST attribute"):
      pipeline_utils.reduce_bipartite_peer_attributes(
          [u0_t0, u0_t1, u0_t2], config, [self.st0, self.st1]
      )


# Tests FeaturizeBipartitePeerAttributes.
# This test fixture controls various scenarios.
# Graph (directed as shown or undirected if direction is unset in proto):
# At snapshot_time st0_:
#   u0 -> cl0 -> u1
#         cl0 -> u2
#   u0 -> cl1 -> u1
#   u1 -> cal0 -> u0
# At snapshot_time st1_:
#   u0 -> cl1 -> u2
#         cl1 -> u3
#   u1 -> cl0 -> u3
#   u1 -> cl1
# Since this library does not control how final weights are distributed among
# peers, the choice of weights are not important as long as they are positive.
# All of the tests of FeaturizeBipartitePeerAttributesTest do not check weights
# of each featurized peer. TwoHopsRandomWalkNeighbors guarantees them.
class FeaturizeBipartitePeerAttributesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Common duration. Not directly relevant to the tests but necessary to make
    # configs.
    self.duration = datetime.timedelta(hours=120)
    self.st0 = time_utils.time_ftz(2023, 8, 2, 4, 0, 0)
    self.st1 = time_utils.time_ftz(2023, 8, 3, 4, 0, 0)

    # PeerAttribute Convention: <name>_<snapshot_time>_<direction>
    self.cl0_st0_f = make_peer_attribute(
        "cl",
        "cl0",
        1.0,
        self.st0,
        context_pb2.PeerAttribute.Direction.D_FORWARD,
    )
    self.cl0_st0_b = make_peer_attribute(
        "cl",
        "cl0",
        2.0,
        self.st0,
        context_pb2.PeerAttribute.Direction.D_BACKWARD,
    )
    self.cl1_st0_f = make_peer_attribute(
        "cl",
        "cl1",
        10.0,
        self.st0,
        context_pb2.PeerAttribute.Direction.D_FORWARD,
    )
    self.cl1_st0_b = make_peer_attribute(
        "cl",
        "cl1",
        20.0,
        self.st0,
        context_pb2.PeerAttribute.Direction.D_BACKWARD,
    )
    self.cal0_st0_f = make_peer_attribute(
        "cal",
        "cal0",
        0.1,
        self.st0,
        context_pb2.PeerAttribute.Direction.D_FORWARD,
    )
    self.cal0_st0_b = make_peer_attribute(
        "cal",
        "cal0",
        0.2,
        self.st0,
        context_pb2.PeerAttribute.Direction.D_BACKWARD,
    )
    self.cl0_st1_f = make_peer_attribute(
        "cl",
        "cl0",
        2.0,
        self.st1,
        context_pb2.PeerAttribute.Direction.D_FORWARD,
    )
    self.cl0_st1_b = make_peer_attribute(
        "cl",
        "cl0",
        4.0,
        self.st1,
        context_pb2.PeerAttribute.Direction.D_BACKWARD,
    )
    self.cl1_st1_f = make_peer_attribute(
        "cl",
        "cl1",
        20.0,
        self.st1,
        context_pb2.PeerAttribute.Direction.D_FORWARD,
    )
    self.cl1_st1_b = make_peer_attribute(
        "cl",
        "cl1",
        40.0,
        self.st1,
        context_pb2.PeerAttribute.Direction.D_BACKWARD,
    )

    # Context convention: <principal>_<name>_<snapshot_time>_<direction>
    # directed context at st0:
    self.u0_cl0_st0_f = make_context("u0", self.st0, [self.cl0_st0_f])
    self.u1_cl0_st0_b = make_context("u1", self.st0, [self.cl0_st0_b])
    self.u2_cl0_st0_b = make_context("u2", self.st0, [self.cl0_st0_b])
    self.u0_cl1_st0_f = make_context("u0", self.st0, [self.cl1_st0_f])
    self.u1_cl1_st0_b = make_context("u1", self.st0, [self.cl1_st0_b])
    self.u1_cal0_st0_f = make_context("u1", self.st0, [self.cal0_st0_f])
    self.u0_cal0_st0_b = make_context("u0", self.st0, [self.cal0_st0_b])
    # directed context at st1:
    self.u0_cl1_st1_f = make_context("u0", self.st1, [self.cl1_st1_f])
    self.u2_cl1_st1_b = make_context("u2", self.st1, [self.cl1_st1_b])
    self.u3_cl1_st1_b = make_context("u3", self.st1, [self.cl1_st1_b])
    self.u1_cl0_st1_f = make_context("u1", self.st1, [self.cl0_st1_f])
    self.u3_cl0_st1_b = make_context("u3", self.st1, [self.cl0_st1_b])
    self.u1_cl1_st1_f = make_context("u1", self.st1, [self.cl1_st1_f])
    # undirected contexts:
    self.u0_cl0_st0_u = make_context_undirected(self.u0_cl0_st0_f)
    self.u1_cl0_st0_u = make_context_undirected(self.u1_cl0_st0_b)
    self.u2_cl0_st0_u = make_context_undirected(self.u2_cl0_st0_b)
    self.u0_cl1_st0_u = make_context_undirected(self.u0_cl1_st0_f)
    self.u1_cl1_st0_u = make_context_undirected(self.u1_cl1_st0_b)
    self.u1_cal0_st0_u = make_context_undirected(self.u1_cal0_st0_f)
    self.u0_cal0_st0_u = make_context_undirected(self.u0_cal0_st0_b)

    # Common configs.
    self.u_config = make_bipartite_graph_config(
        [context_source_config_pb2.BipartiteGraph.TraversalMode.TM_UNDIRECTED],
        self.duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    self.f_config = make_bipartite_graph_config(
        [context_source_config_pb2.BipartiteGraph.TraversalMode.TM_FORWARD],
        self.duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    self.b_config = make_bipartite_graph_config(
        [context_source_config_pb2.BipartiteGraph.TraversalMode.TM_BACKWARD],
        self.duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    self.fb_config = make_bipartite_graph_config(
        [
            context_source_config_pb2.BipartiteGraph.TraversalMode.TM_FORWARD,
            context_source_config_pb2.BipartiteGraph.TraversalMode.TM_BACKWARD,
        ],
        self.duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    self.fbu_config = make_bipartite_graph_config(
        [
            context_source_config_pb2.BipartiteGraph.TraversalMode.TM_FORWARD,
            context_source_config_pb2.BipartiteGraph.TraversalMode.TM_BACKWARD,
            context_source_config_pb2.BipartiteGraph.TraversalMode.TM_UNDIRECTED,
        ],
        self.duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )
    self.un_config = make_bipartite_graph_config(
        [context_source_config_pb2.BipartiteGraph.TraversalMode.TM_UNSET],
        self.duration,
        context_source_config_pb2.BipartiteGraph.EWM_LATEST,
    )

    self.agg_method = (
        context_source_config_pb2.PeerFeatureConfig.AttributeValueAggregationMethod.AGG_ACCUMULATE
    )

  def test_empty_on_empty(self):
    pfc = make_peer_feature_config("cl", 10, self.u_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])
    res = pipeline_utils.featurize_bipartite_peer_attributes([], config)
    self.assertEqual(len(res), 0)

  # The following tests only uses the graph of
  #   u0 -> cl0 -> u1
  #         cl0 -> u2
  def test_works_forward(self):
    pfc = make_peer_feature_config("cl", 10, self.f_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    # u1 and u2 are featurized by u0.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [self.u0_cl0_st0_f, self.u1_cl0_st0_b, self.u2_cl0_st0_b], config
    )

    u1_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    u2_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    self.assertEqual(len(res), 2)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    self.assertIn("u1", res_dict)
    assert_proto2_equal_ignoring_fields(
        u1_context,
        res_dict["u1"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
    )
    self.assertIn("u2", res_dict)
    assert_proto2_equal_ignoring_fields(
        u2_context,
        res_dict["u2"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
    )

  def test_works_backward(self):
    pfc = make_peer_feature_config("cl", 10, self.b_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    # u0 is featurized by u1 and u2.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [self.u0_cl0_st0_f, self.u1_cl0_st0_b, self.u2_cl0_st0_b], config
    )

    u0_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_b"
            bag_of_weighted_words {
              tokens { token: "u1" }
              tokens { token: "u2" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    self.assertEqual(len(res), 1)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    self.assertIn("u0", res_dict)

    assert_proto2_equal_ignoring_fields(
        u0_context,
        res_dict["u0"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=["peer_attributes"],
    )

  def test_works_undirected(self):
    pfc = make_peer_feature_config("cl", 10, self.u_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    # u0, u1, u2 are all connected by all via two hops, including themselves.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [self.u0_cl0_st0_u, self.u1_cl0_st0_u, self.u2_cl0_st0_u], config
    )

    expected_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_u"
            bag_of_weighted_words {
              tokens { token: "u0" }
              tokens { token: "u1" }
              tokens { token: "u2" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    self.assertEqual(len(res), 3)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    for key in ["u0", "u1", "u2"]:
      self.assertIn(key, res_dict)
      self.assertEqual(len(res_dict[key]), 1)
      assert_proto2_equal_ignoring_fields(
          expected_context,
          res_dict[key][0],
          ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
          ignore_order_in_fields=[
              "features_per_source.features.bag_of_weighted_words.tokens"
          ],
      )

  def test_works_forward_backward(self):
    pfc = make_peer_feature_config("cl", 10, self.fb_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    # forward case: u1 and u2 are featurized by u0 with "_f" appended.
    # backward case: u0 is featurized by u1 and u2 with "_b" appended.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [self.u0_cl0_st0_f, self.u1_cl0_st0_b, self.u2_cl0_st0_b], config
    )

    u1_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    u2_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    u0_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_b"
            bag_of_weighted_words {
              tokens { token: "u1" }
              tokens { token: "u2" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    self.assertEqual(len(res), 3)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    # Check u0
    self.assertIn("u0", res_dict)
    self.assertEqual(len(res_dict["u0"]), 1)
    assert_proto2_equal_ignoring_fields(
        u0_context,
        res_dict["u0"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

    # Check u1
    self.assertIn("u1", res_dict)
    self.assertEqual(len(res_dict["u1"]), 1)
    assert_proto2_equal_ignoring_fields(
        u1_context,
        res_dict["u1"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

    # Check u2
    self.assertIn("u2", res_dict)
    self.assertEqual(len(res_dict["u2"]), 1)
    assert_proto2_equal_ignoring_fields(
        u2_context,
        res_dict["u2"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

  def test_works_forward_backward_undirected(self):
    pfc = make_peer_feature_config("cl", 10, self.fbu_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    # forward case: u1 and u2 are featurized by u0 with "_f" appended.
    # backward case: u0 is featurized by u1 and u2 with "_b" appended.
    # undirected case: everybody is featurized by everybody with "_u" appended.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [self.u0_cl0_st0_f, self.u1_cl0_st0_b, self.u2_cl0_st0_b], config
    )

    u1_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    u2_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    u0_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_b"
            bag_of_weighted_words {
              tokens { token: "u1" }
              tokens { token: "u2" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    undirected_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_u"
            bag_of_weighted_words {
              tokens { token: "u0" }
              tokens { token: "u1" }
              tokens { token: "u2" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    self.assertEqual(len(res), 6)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    expected_contexts = {
        "u0": [u0_context, undirected_context],
        "u1": [u1_context, undirected_context],
        "u2": [u2_context, undirected_context],
    }

    for key in ["u0", "u1", "u2"]:
      self.assertIn(key, res_dict)
      self.assertEqual(len(res_dict[key]), 2)
      # Sort by feature name to have deterministic order
      res_dict[key].sort(
          key=lambda x: x.features_per_source[0].features[0].name
      )
      expected_contexts[key].sort(
          key=lambda x: x.features_per_source[0].features[0].name
      )

      assert_proto2_equal_ignoring_fields(
          expected_contexts[key][0],
          res_dict[key][0],
          ignored_fields=[
              "features_per_source.features.bag_of_weighted_words.tokens.weight"
          ],
          ignore_order_in_fields=[
              "features_per_source.features.bag_of_weighted_words.tokens"
          ],
      )
      assert_proto2_equal_ignoring_fields(
          expected_contexts[key][1],
          res_dict[key][1],
          ignored_fields=[
              "features_per_source.features.bag_of_weighted_words.tokens.weight"
          ],
          ignore_order_in_fields=[
              "features_per_source.features.bag_of_weighted_words.tokens"
          ],
      )

  def test_removes_zero_weight_backward(self):
    pfc = make_peer_feature_config("cl", 10, self.b_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    u2_cl0_st0_b_zero = context_pb2.Context()
    u2_cl0_st0_b_zero.CopyFrom(self.u2_cl0_st0_b)
    u2_cl0_st0_b_zero.peer_attributes[0].weight = 0.0
    # u0 is featurized by u1 since u2's weight with cl0 is zero.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [self.u0_cl0_st0_f, self.u1_cl0_st0_b, u2_cl0_st0_b_zero], config
    )

    u0_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    self.assertEqual(len(res), 1)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    self.assertIn("u0", res_dict)
    assert_proto2_equal_ignoring_fields(
        u0_context,
        res_dict["u0"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=["peer_attributes"],
    )

  def test_removes_zero_weight_forward(self):
    pfc = make_peer_feature_config("cl", 10, self.f_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    u0_cl0_st0_f_zero = context_pb2.Context()
    u0_cl0_st0_f_zero.CopyFrom(self.u0_cl0_st0_f)
    u0_cl0_st0_f_zero.peer_attributes[0].weight = 0.0
    # Since u0 is not connected with cl0, there is no featurization.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [u0_cl0_st0_f_zero, self.u1_cl0_st0_b, self.u2_cl0_st0_b], config
    )

    self.assertEqual(len(res), 0)

  # The following test uses the graph.
  #   u0 -> cl1 -> u2
  #         cl1 -> u3
  #   u1 -> cl0 -> u3
  def test_works_two_attributes(self):
    pfc = make_peer_feature_config("cl", 10, self.f_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    # u2 is featurized by u0. u3 is featurized by u0 and u1.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [
            self.u0_cl1_st1_f,
            self.u2_cl1_st1_b,
            self.u3_cl1_st1_b,
            self.u1_cl0_st1_f,
            self.u3_cl0_st1_b,
        ],
        config,
    )

    u2_context = text_format.Parse(
        """
        valid_from { seconds: 1691035200 }
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

    u3_context = text_format.Parse(
        """
        valid_from { seconds: 1691035200 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_f"
            bag_of_weighted_words {
              tokens { token: "u0" }
              tokens { token: "u1" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    self.assertEqual(len(res), 2)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    # Check u2
    self.assertIn("u2", res_dict)
    self.assertEqual(len(res_dict["u2"]), 1)
    assert_proto2_equal_ignoring_fields(
        u2_context,
        res_dict["u2"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

    # Check u3
    self.assertIn("u3", res_dict)
    self.assertEqual(len(res_dict["u3"]), 1)
    assert_proto2_equal_ignoring_fields(
        u3_context,
        res_dict["u3"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

  # The following test uses the graph with possible config tweaking.
  #   u0 -> cl0 -> u1
  #         cl0 -> u2
  #   u1 -> cal0 -> u0
  def test_works_bipartite_graph_filtering(self):
    pfc_cl = make_peer_feature_config("cl", 10, self.b_config, self.agg_method)
    pfc_cal = make_peer_feature_config(
        "cal", 10, self.u_config, self.agg_method
    )
    pfc_cal.ClearField("bipartite_graph")
    config = make_context_source_config(self.duration, [pfc_cl, pfc_cal])

    # u0 is featurized by u1 and u2.
    # u1 is not featurized by u0 since no bipartite config is present for "cal".
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [
            self.u0_cl0_st0_f,
            self.u1_cl0_st0_b,
            self.u2_cl0_st0_b,
            self.u1_cal0_st0_u,
            self.u0_cal0_st0_u,
        ],
        config,
    )

    u0_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_b"
            bag_of_weighted_words {
              tokens { token: "u1" }
              tokens { token: "u2" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    self.assertEqual(len(res), 1)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    self.assertIn("u0", res_dict)
    assert_proto2_equal_ignoring_fields(
        u0_context,
        res_dict["u0"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=["peer_attributes"],
    )

  # The following test uses the graph.
  #   u0 -> cl1 -> u1
  #   u1 -> cal0 -> u0
  def test_works_two_attribute_names(self):
    pfc_cl = make_peer_feature_config("cl", 10, self.f_config, self.agg_method)
    pfc_cal = make_peer_feature_config(
        "cal", 10, self.f_config, self.agg_method
    )
    config = make_context_source_config(self.duration, [pfc_cl, pfc_cal])

    # cl case: u1 is featurized by u0.
    # cal case: u0 is featurized by u1.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [
            self.u0_cl1_st0_f,
            self.u1_cl1_st0_b,
            self.u1_cal0_st0_f,
            self.u0_cal0_st0_b,
        ],
        config,
    )

    u1_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    u0_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cal_f"
            bag_of_weighted_words { tokens { token: "u1" } }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    self.assertEqual(len(res), 2)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    self.assertIn("u0", res_dict)
    assert_proto2_equal_ignoring_fields(
        u0_context,
        res_dict["u0"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=["peer_attributes"],
    )
    self.assertIn("u1", res_dict)
    assert_proto2_equal_ignoring_fields(
        u1_context,
        res_dict["u1"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=["peer_attributes"],
    )

  # The following test uses the graph of
  # At snapshot_time st0_:
  #   u0 -> cl0 -> u1
  #         cl0 -> u2
  #   u0 -> cl1 -> u1
  # At snapshot_time st1_:
  #   u0 -> cl1 -> u2
  #   u1 -> cl0 -> u3
  #   u1 -> cl1
  def test_works_two_snapshot_times(self):
    pfc = make_peer_feature_config("cl", 10, self.f_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])
    # At st0_: u1 an u2 are featurized by u0.
    # At st1_: u2 is featurized by u0 and u1. u3 is featurized by u1 only.

    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [
            self.u0_cl0_st0_f,
            self.u1_cl0_st0_b,
            self.u2_cl0_st0_b,
            self.u0_cl1_st0_f,
            self.u1_cl1_st0_b,
            self.u0_cl1_st1_f,
            self.u2_cl1_st1_b,
            self.u1_cl0_st1_f,
            self.u3_cl0_st1_b,
            self.u1_cl1_st1_f,
        ],
        config,
    )

    u1_context_st0 = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    u2_context_st0 = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
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

    u2_context_st1 = text_format.Parse(
        """
        valid_from { seconds: 1691035200 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_f"
            bag_of_weighted_words {
              tokens { token: "u0" }
              tokens { token: "u1" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    u3_context_st1 = text_format.Parse(
        """
        valid_from { seconds: 1691035200 }
        features_per_source {
          source_type: "test"
          features {
            name: "cl_f"
            bag_of_weighted_words { tokens { token: "u1" } }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    expected = {
        "u1": [u1_context_st0],
        "u2": [u2_context_st0, u2_context_st1],
        "u3": [u3_context_st1],
    }

    self.assertEqual(len(res), 4)  # (u1, st0), (u2, st0), (u2, st1), (u3, st1)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    self.assertNotIn("u0", res_dict)

    # Check u1
    self.assertIn("u1", res_dict)
    self.assertEqual(len(res_dict["u1"]), 1)
    assert_proto2_equal_ignoring_fields(
        expected["u1"][0],
        res_dict["u1"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

    # Check u2
    self.assertIn("u2", res_dict)
    self.assertEqual(len(res_dict["u2"]), 2)
    # Sort contexts by valid_from to match expected order
    res_dict["u2"].sort(key=lambda x: x.valid_from.seconds)
    assert_proto2_equal_ignoring_fields(
        expected["u2"][0],
        res_dict["u2"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )
    assert_proto2_equal_ignoring_fields(
        expected["u2"][1],
        res_dict["u2"][1],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

    # Check u3
    self.assertIn("u3", res_dict)
    self.assertEqual(len(res_dict["u3"]), 1)
    assert_proto2_equal_ignoring_fields(
        expected["u3"][0],
        res_dict["u3"][0],
        ["features_per_source.features.bag_of_weighted_words.tokens.weight"],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

  # The following test uses the graph.
  #   u0 - u0 - u1
  #   u2 - u1 - u3
  # Undirected graphs where "u0" and "u2" in the left and "u1" and "u3" in the
  # right are principals. "u0" and "u1" in the middle are attributes.
  def test_works_principal_equals_to_attribute(self):
    pfc = make_peer_feature_config("cal", 10, self.u_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])

    calu0 = make_peer_attribute(
        "cal", "u0", 1.0, self.st0, context_pb2.PeerAttribute.Direction.D_UNSET
    )
    u0_cal0 = make_context("u0", self.st0, [calu0])
    u1_cal0 = make_context("u1", self.st0, [calu0])
    calu1 = make_peer_attribute(
        "cal", "u1", 2.0, self.st0, context_pb2.PeerAttribute.Direction.D_UNSET
    )
    u2_cal1 = make_context("u2", self.st0, [calu1])
    u3_cal1 = make_context("u3", self.st0, [calu1])

    # u0 and u1 are featurized by u1 and u0.
    # u2 and u3 are featurized by u3 and u2.
    res = pipeline_utils.featurize_bipartite_peer_attributes(
        [u0_cal0, u1_cal0, u2_cal1, u3_cal1], config
    )

    u0_cal0_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cal_u"
            bag_of_weighted_words {
              tokens { token: "u0" }
              tokens { token: "u1" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    u2_cal1_context = text_format.Parse(
        """
        valid_from { seconds: 1690948800 }
        features_per_source {
          source_type: "test"
          features {
            name: "cal_u"
            bag_of_weighted_words {
              tokens { token: "u2" }
              tokens { token: "u3" }
            }
          }
        }
        """,
        contextualized_actions_pb2.FeaturizedContext(),
    )

    self.assertEqual(len(res), 4)

    res_dict = collections.defaultdict(list)
    for key, value in res:
      res_dict[key].append(value)

    # Check u0
    self.assertIn("u0", res_dict)
    self.assertEqual(len(res_dict["u0"]), 1)
    assert_proto2_equal_ignoring_fields(
        u0_cal0_context,
        res_dict["u0"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

    # Check u1
    self.assertIn("u1", res_dict)
    self.assertEqual(len(res_dict["u1"]), 1)
    assert_proto2_equal_ignoring_fields(
        u0_cal0_context,
        res_dict["u1"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

    # Check u2
    self.assertIn("u2", res_dict)
    self.assertEqual(len(res_dict["u2"]), 1)
    assert_proto2_equal_ignoring_fields(
        u2_cal1_context,
        res_dict["u2"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

    # Check u3
    self.assertIn("u3", res_dict)
    self.assertEqual(len(res_dict["u3"]), 1)
    assert_proto2_equal_ignoring_fields(
        u2_cal1_context,
        res_dict["u3"][0],
        ignored_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens.weight"
        ],
        ignore_order_in_fields=[
            "features_per_source.features.bag_of_weighted_words.tokens"
        ],
    )

  def test_dies_on_unset_traversal(self):
    pfc = make_peer_feature_config("cl", 10, self.un_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])
    with self.assertRaisesRegex(ValueError, "Unsupported traversal mode"):
      pipeline_utils.featurize_bipartite_peer_attributes(
          [self.u0_cl0_st0_u], config
      )

  def test_dies_on_forward_unset(self):
    # TM_FORWARD config.
    pfc = make_peer_feature_config("cl", 10, self.f_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])
    # D_UNSET context.
    with self.assertRaisesRegex(ValueError, "incompatible traversal mode"):
      pipeline_utils.featurize_bipartite_peer_attributes(
          [self.u0_cl0_st0_u], config
      )

  def test_dies_on_backward_unset(self):
    # TM_BACKWARD config.
    pfc = make_peer_feature_config("cl", 10, self.b_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])
    # D_UNSET context.
    with self.assertRaisesRegex(ValueError, "incompatible traversal mode"):
      pipeline_utils.featurize_bipartite_peer_attributes(
          [self.u1_cl0_st0_u], config
      )

  def test_dies_on_multiple_peer_attributes(self):
    pfc = make_peer_feature_config("cl", 10, self.u_config, self.agg_method)
    config = make_context_source_config(self.duration, [pfc])
    # Context of two peer attributes.
    u0_st0_f = make_context("u0", self.st0, [self.cl0_st0_f, self.cl1_st0_f])
    with self.assertRaisesRegex(
        ValueError, "Context must have only one peer_attribute"
    ):
      pipeline_utils.featurize_bipartite_peer_attributes([u0_st0_f], config)


if __name__ == "__main__":
  unittest.main()

