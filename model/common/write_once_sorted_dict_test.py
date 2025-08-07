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
from model.common import write_once_sorted_dict as wosd


class WriteOnceSortedDictTest(unittest.TestCase):

  def test_sorted(self):
    d = wosd.WriteOnceSortedDict()
    d['d'] = 0
    d['b'] = 1
    d['a'] = 2
    d['c'] = 3
    self.assertEqual(list(d.items()), [('a', 2), ('b', 1), ('c', 3), ('d', 0)])
    self.assertEqual(list(d.values()), [2, 1, 3, 0])

  def test_set_existing(self):
    d = wosd.WriteOnceSortedDict()
    d['key'] = 0
    with self.assertRaises(ValueError):
      d['key'] = 0
    self.assertEqual(d['key'], 0)

  def test_update_existing(self):
    d = wosd.WriteOnceSortedDict()
    d['key'] = 0
    with self.assertRaises(ValueError):
      d.update({'key': 0})
    self.assertEqual(d['key'], 0)

  def test_deletion(self):
    d = wosd.WriteOnceSortedDict()
    d['key'] = 0
    with self.assertRaises(ValueError):
      del d['key']


if __name__ == '__main__':
  unittest.main()
