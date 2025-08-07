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
from protos import duration_pb2
from protos import timestamp_pb2


class TimeUtilsTest(unittest.TestCase):

  def test_timestamp_conversion(self):
    proto_t = timestamp_pb2.Timestamp(seconds=665553906, nanos=0)
    py_t = time_utils.convert_proto_time_to_time(proto_t)
    proto_t2 = time_utils.convert_to_proto_time(py_t)
    py_t2 = time_utils.convert_proto_time_to_time(proto_t2)
    self.assertEqual(proto_t, proto_t2)
    self.assertEqual(py_t, py_t2)


  def test_duration_conversion(self):
    proto_d = duration_pb2.Duration(seconds=301, nanos=0)
    py_d = time_utils.convert_proto_duration_to_timedelta(proto_d)
    proto_d2 = time_utils.convert_to_proto_duration(py_d)
    py_d2 = time_utils.convert_proto_duration_to_timedelta(proto_d2)
    self.assertEqual(proto_d, proto_d2)
    self.assertEqual(py_d, py_d2)


class GenerateSnapshotTimesTest(unittest.TestCase):

  def test_empty_on_end_before_start(self):
    tz = time_utils.get_facade_timezone()
    start = datetime.datetime.fromtimestamp(1, tz=tz)
    period = datetime.timedelta(hours=1)
    end_eq = datetime.datetime.fromtimestamp(1, tz=tz)
    self.assertEqual([], time_utils.generate_snapshot_times(start, end_eq, period))

    end_lt = datetime.datetime.fromtimestamp(0, tz=tz)
    self.assertEqual([], time_utils.generate_snapshot_times(start, end_lt, period))


  def test_returns_empty_when_no_time_is_viable(self):
    tz = time_utils.get_facade_timezone()
    start = datetime.datetime.fromtimestamp(1, tz=tz)
    end = datetime.datetime.fromtimestamp(10, tz=tz)
    period = datetime.timedelta(hours=1)
    offset = datetime.timedelta(seconds=10)
    # First available time is 10 seconds from Unix zero second.
    self.assertEqual([], time_utils.generate_snapshot_times(start, end, period, offset))


  def test_works_when_start_time_is_aligned(self):
    tz = time_utils.get_facade_timezone()
    # start = 2023-10-27 06:00:00 UTC
    start = datetime.datetime.fromtimestamp(1698386400, tz=tz)
    # end = 2023-10-27 22:00:00 UTC
    end = datetime.datetime.fromtimestamp(1698444000, tz=tz)
    offset = datetime.timedelta(hours=2)
    period = datetime.timedelta(hours=4)

    result = time_utils.generate_snapshot_times(start, end, period, offset)
        
    # Expected times:
    # t1 = 06:00:00 (start time)
    # t2 = 10:00:00
    # t3 = 14:00:00
    # t4 = 18:00:00
    t1 = start
    t2 = t1 + period
    t3 = t2 + period
    t4 = t3 + period
    expected = [t1, t2, t3, t4]

    self.assertEqual(expected, result)


  def test_works_when_start_time_is_not_aligned(self):
    tz = time_utils.get_facade_timezone()
    # start = 2023-10-27 03:00:00 UTC
    start = datetime.datetime.fromtimestamp(1698375600, tz=tz)
    # end = 2023-10-27 22:00:00 UTC
    end = datetime.datetime.fromtimestamp(1698444000, tz=tz)
    offset = datetime.timedelta(hours=2)
    period = datetime.timedelta(hours=4)

    result = time_utils.generate_snapshot_times(start, end, period, offset)
        
    # Expected times:
    # First viable time is at 06:00:00 (start + 3 hours)
    # t1 = 06:00:00
    # t2 = 10:00:00
    # t3 = 14:00:00
    # t4 = 18:00:00
    t1 = start + datetime.timedelta(hours=3)
    t2 = t1 + period
    t3 = t2 + period
    t4 = t3 + period
    expected = [t1, t2, t3, t4]

    self.assertEqual(expected, result)
    

if __name__ == '__main__':
  unittest.main()
