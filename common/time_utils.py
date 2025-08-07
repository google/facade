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
"""Utility functions for dealing with time in Facade."""

import datetime
from typing import List

from protos import duration_pb2
from protos import timestamp_pb2


def parse_datetime_flag(flag: str) -> datetime.datetime:
  naive_dt = datetime.datetime.strptime(flag, '%Y-%m-%d %H:%M:%S')
  return naive_dt.replace(tzinfo=get_facade_timezone())


def get_facade_timezone() -> datetime.tzinfo:
  """Gets Facade timezone (UTC)."""
  return datetime.timezone.utc


def time_ftz(year: int, month: int, day: int,
             hour: int, minute: int, second: int) -> datetime.datetime:
  """
  Absolute time from civil time interpreted in the facade's timezone (UTC).
  """
  return datetime.datetime(year, month, day, hour, minute, second,
                           tzinfo=get_facade_timezone())


def convert_to_proto_time(t: datetime.datetime) -> timestamp_pb2.Timestamp:
  return timestamp_pb2.Timestamp(
      seconds=int(t.timestamp()),
      nanos=int(t.microsecond * 1000)
  )


def convert_proto_time_to_time(t: timestamp_pb2.Timestamp) -> datetime.datetime:
  full_timestamp = t.seconds + (t.nanos / 1_000_000_000)
  return datetime.datetime.fromtimestamp(full_timestamp,
                                         tz=datetime.timezone.utc)


def convert_to_proto_duration(d: datetime.timedelta) -> duration_pb2.Duration:
  return duration_pb2.Duration(
      seconds=int(d.total_seconds()),
      nanos=int(d.microseconds * 1000),
  )


def convert_proto_duration_to_timedelta(d: duration_pb2.Duration) -> datetime.timedelta:
  micros = d.nanos // 1000
  return datetime.timedelta(seconds=d.seconds, microseconds=micros)


def generate_snapshot_times(
    start: datetime.datetime,
    end: datetime.datetime,
    period: datetime.timedelta,
    offset: datetime.timedelta = datetime.timedelta()
) -> List[datetime.datetime]:
    """
    Generates times separated by period that fall in [start, end).

    Start from offset from midnight in the Facade time zone.
    Example (all times in FacadeTimeZone):
      start = 2023-10-27T03:00:00, end = 2023-10-27T22:00:00, offset = 2h, period = 2h.
    Returns: 2023-10-27T04:00:00, 2023-10-27T06:00:00 ..., 2023-10-27T20:00:00.

    Returns empty when start >= end or no available time by offset and period.
    from offset from midnight.
    Generates a series of timestamps based on a start/end time and a period.

    Args:
        start: The start time for the generation period (timezone-aware).
        end: The end time for the generation period (timezone-aware).
        period: The interval between each generated timestamp.
        offset: The duration from the start of the day to the first potential time.

    Returns:
        A list of generated datetime objects.
    """
    if start >= end:
        return []
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("The 'start' datetime must be timezone-aware.")

    start_of_day_utc = datetime.datetime(
        start.year, start.month, start.day, tzinfo=get_facade_timezone()
    )
    current_time = start_of_day_utc + offset
    while current_time < start:
        current_time += period
    result = []
    while current_time < end:
        result.append(current_time)
        current_time += period
    return result
