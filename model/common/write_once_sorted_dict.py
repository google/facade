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
"""A write-once ordered dictionary."""

from collections.abc import MutableMapping
from typing import Generic, TypeVar

import sortedcontainers

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.


class WriteOnceSortedDict(MutableMapping, Generic[KT, VT]):
  """Sorted dictionary where keys can be assigned at most once."""

  def __init__(self, *args, **kwds):
    self._dict = sortedcontainers.SortedDict(*args, **kwds)

  def __setitem__(self, k: KT, v: VT):
    if k in self._dict:
      raise ValueError('Attempting to set value for existing key: %r' % k)
    else:
      self._dict.__setitem__(k, v)

  def __getitem__(self, *args, **kwargs):
    return self._dict.__getitem__(*args, **kwargs)

  def __delitem__(self, *args, **kwargs):
    raise ValueError('Deletion not supported')

  def __iter__(self, *args, **kwargs):
    return self._dict.__iter__(*args, **kwargs)

  def __len__(self, *args, **kwargs):
    return self._dict.__len__(*args, **kwargs)
