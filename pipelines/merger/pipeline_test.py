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

from common import time_utils
from pipelines.merger import pipeline
from protos import contextualized_actions_pb2


def make_fa(t):
  return contextualized_actions_pb2.FeaturizedAction(
        occurred_at=time_utils.convert_to_proto_time(t))


def make_fabs(s, fas):
  return contextualized_actions_pb2.FeaturizedActionsBySource(
        source_type=s, actions=fas)


def make_fc(t, s):
  return contextualized_actions_pb2.FeaturizedContext(
        valid_from=time_utils.convert_to_proto_time(t),
        features_per_source=[contextualized_actions_pb2.ContextSourceFeatures(
            source_type=s
        )])


def make_ca(p, c, a):
  return contextualized_actions_pb2.ContextualizedActions(
      principal=p,
      context=c,
      actions=a
  )


class PipelineTest(unittest.TestCase):

  def test_ignores_uninitializable_actions(self):
    past = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    future = time_utils.time_ftz(2023, 1, 2, 12, 0, 0)
    fa = make_fa(past)
    fabs = make_fabs('source', [fa])
    fc = make_fc(future, 'context_source')

    # No context is available at action time.
    res = pipeline.contextualize_actions(
        [('p', fc)],
        {'source': [('p', fa)]},
        [future],
        10)
    # Only context is possible at "future".
    expected = make_ca("p", fc, []);
    self.assertEqual(res, [expected])


  def test_works_one_action(self):
    past = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    future = time_utils.time_ftz(2023, 1, 2, 11, 0, 0)
    fa = make_fa(future)
    fabs = make_fabs("source", [fa])
    fc = make_fc(past, "context_source")

    res = pipeline.contextualize_actions(
        [('p', fc)],
        {'source': [('p', fa)]},
        [past], 10)
    expected = make_ca('p', fc, [fabs])
    self.assertEqual(res, [expected])


  def test_drops_actions_only(self):
    past = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    future = time_utils.time_ftz(2023, 1, 2, 11, 0, 0)
    fa = make_fa(future)
    fabs = make_fabs("source", [fa])
    fc = make_fc(past, "context_source")

    res = pipeline.contextualize_actions(
        [], {"source": [('p', fa)]}, [past], 10)
    self.assertEqual(res, [])


  def test_works_context_only(self):
    past = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    future = time_utils.time_ftz(2023, 1, 2, 11, 0, 0)
    fa = make_fa(future)
    fabs = make_fabs("source", [fa])
    fc = make_fc(past, "context_source")

    res = pipeline.contextualize_actions(
        [('p', fc)],
        {},
        [past], 10)

    expected = make_ca("p", fc, [fabs])
    del expected.actions[:]
    self.assertEqual(res, [expected])


  def test_works_multiple_action_sources(self):
    t0 = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    t1 = time_utils.time_ftz(2023, 1, 2, 1, 0, 0)
    t2 = time_utils.time_ftz(2023, 1, 2, 11, 0, 0)
    fa0 = make_fa(t0)
    fa1 = make_fa(t1)
    fa2 = make_fa(t2)
    fabs1 = make_fabs("source_1", [fa0, fa1])
    fabs2 = make_fabs("source_2", [fa1, fa2])
    fc = make_fc(t0, "context_source")

    # Regardless of the order of the input actions, the final result must be
    # sorted.
    res_1_2 = pipeline.contextualize_actions(
        [('p', fc)],
        {'source_1': [('p', fa0), ('p', fa1)],
         'source_2': [('p', fa1), ('p', fa2)]},
        [t0], 10)
    res_2_1 = pipeline.contextualize_actions(
        [('p', fc)],
        {'source_2': [('p', fa1), ('p', fa2)],
         'source_1': [('p', fa0), ('p', fa1)]},
        [t0], 10)

    expected = make_ca('p', fc, [fabs1, fabs2])
    self.assertEqual(res_1_2, [expected])
    self.assertEqual(res_2_1, [expected])


  def test_works_multiple_principals(self):
    t0 = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    t1 = time_utils.time_ftz(2023, 1, 2, 1, 0, 0)
    t2 = time_utils.time_ftz(2023, 1, 2, 11, 0, 0)
    fa0 = make_fa(t0)
    fa1 = make_fa(t1)
    fa2 = make_fa(t2)
    fabs1 = make_fabs("source_1", [fa0, fa1])
    fabs2 = make_fabs("source_2", [fa0, fa1, fa2])
    fc0 = make_fc(t0, "context_source")
    fc1 = make_fc(t1, "context_source")
    fc2 = make_fc(t2, "context_source")

    # No action by context "p2". "p0" principals have two timestamps to match
    # fa0 and fa1, respectively. fa0 and fa2 for context "p1" is not available
    # since "p1" does not have context at t0 and t2.
    res = pipeline.contextualize_actions(
        [('p0', fc0), ('p0', fc1), ('p1', fc1), ('p2', fc2)],
        {"source_1": [("p0", fa0), ("p0", fa1)],
         "source_2": [("p1", fa0), ("p1", fa1), ("p1", fa2)]},
        [t0, t1, t2], 10)

    expected0 = make_ca("p0", fc0, [make_fabs("source_1", [fa0])]);
    expected1 = make_ca("p0", fc1, [make_fabs("source_1", [fa1])]);
    expected2 = make_ca("p1", fc1, [make_fabs("source_2", [fa1])]);
    # "p2" is context only.
    expected3 = make_ca("p2", fc2, []);
    # "p1" at t0 is action only. Dropped.
    # "p1" at t2 is action only. Dropped.
    self.assertEqual(res, [expected0, expected1, expected2, expected3])
    

  def test_works_more_than_max_actions(self):
    t0 = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    t1 = time_utils.time_ftz(2023, 1, 2, 1, 0, 0)
    t2 = time_utils.time_ftz(2023, 1, 2, 11, 0, 0)
    fa0 = make_fa(t0)
    fa1 = make_fa(t1)
    fa2 = make_fa(t2)
    fabs = make_fabs("source", [fa0, fa1, fa2])
    fc = make_fc(t0, "context_source")

    # There are 3 actions. Two ContextualizedActions are emitted where one with
    # 2 actions and one with 1 action.
    res = pipeline.contextualize_actions(
        [('p', fc)],
        {'source': [('p', fa0), ('p', fa1), ('p', fa2)]},
        [t0], 2)

    # We cannot guarantee which actions are emitted in which contextualized
    # actions. So we inspect the contents.
    res_actions = []
    ca_counter = 0
    for value in res:
      self.assertEqual(value.principal, 'p')
      self.assertEqual(value.context, fc)
      self.assertEqual(len(value.actions), 1)
      self.assertEqual(value.actions[0].source_type, 'source')
      for action in value.actions[0].actions:
        res_actions.append(action)
      ca_counter += 1
    self.assertEqual(ca_counter, 2)
    self.assertEqual(res_actions, [fa0, fa1, fa2])


  def test_works_multiple_incomplete_sources(self):
    t0 = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    t1 = time_utils.time_ftz(2023, 1, 2, 1, 0, 0)
    t2 = time_utils.time_ftz(2023, 1, 2, 11, 0, 0)
    fa0 = make_fa(t0)
    fa1 = make_fa(t1)
    fa2 = make_fa(t2)
    fabs = make_fabs("source", [fa0, fa1, fa2])
    fc0 = make_fc(t0, "cs1")
    fc01 = make_fc(t1, "cs1")
    fc1 = make_fc(t1, "cs2")
    fc2 = make_fc(t2, "cs2")

    # At t0, context from "cs1" source is available. At t1, context from both
    # "cs1" and "cs2" are available. At t2, context from "cs2" is available.
    res = pipeline.contextualize_actions(
        [('p', fc0), ('p', fc01), ('p', fc1), ('p', fc2)],
        {'source': [('p', fa0), ('p', fa1), ('p', fa2)]},
        [t0, t1, t2], 10)

    expected0 = make_ca("p", fc0, [make_fabs("source", [fa0])])
    expected1 = make_ca("p", fc01, [make_fabs("source", [fa1])])
    expected1.context.features_per_source.add(source_type="cs2")
    expected2 = make_ca("p", fc2, [make_fabs("source", [fa2])]);
    self.assertEqual(res, [expected0, expected1, expected2])


  def works_ca_limit(self):
    t0 = time_utils.time_ftz(2023, 1, 1, 12, 0, 0)
    t1 = time_utils.time_ftz(2023, 1, 2, 1, 0, 0)
    t2 = time_utils.time_ftz(2023, 1, 2, 11, 0, 0)
    fa0 = make_fa(t0)
    fa1 = make_fa(t1)
    fa2 = make_fa(t2)
    fabs = make_fabs("source", [fa0, fa1, fa2])
    fc = make_fc(t0, "context_source")

    # There are 3 actions. Two ContextualizedActions are emitted where one with
    # 2 actions and one with 1 action. Also, the number of maximum
    # contextualized actions per principal and snapshot time is set to 1.
    res = pipeline.contextualize_actions(
        [('p', fc)],
        {'source', [('p', fa0), ('p', fa1), ('p', fa2)]},
        [t0], 2, 1)

    # We cannot guarantee which actions are emitted in the contextualized
    # actions. So we inspect the numbers only.
    res_actions = []
    ca_counter = 0
    for value in res:
      self.assertEqual(value.principal, 'p')
      self.assertEqual(value.context, fc)
      self.assertEqual(len(value.actions), 1)
      self.assertEqual(value.actions[0].source_type, 'source')
      for action in value.actions[0].actions:
        res_actions.append(action)
      ca_counter += 1
    self.assertEqual(ca_counter, 1)
    self.assertEqual(len(res_actions), 2)
    
    
if __name__ == '__main__':
  unittest.main()
