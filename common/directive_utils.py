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
"""Utility function for reading serialized directive from disk."""

from google.protobuf import text_format
from protos import directive_pb2


def read_directive(textproto_filename: str) -> directive_pb2.Directive:
  """Reads a textproto-serialized directive file."""
  config = directive_pb2.Directive()
  with open(textproto_filename, 'r') as f:
    text_format.Parse(f.read(), config)
  return config
