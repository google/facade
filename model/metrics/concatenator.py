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
"""Concatenate values across batches for exact large-scale data measurements."""

from collections.abc import Iterable
from typing import Any

import tensorflow as tf
from tensorflow import keras


# Growth factor between reallocations of the dynamic array.
_GROWTH_FACTOR = 1.2


def _next_array_size(minimum_size: tf.Tensor) -> tf.Tensor:
  """Calculate the next size for reallocating a dynamic array.

  Args:
    minimum_size: A scalar tensor representing the minimal required array
      capacity. Expects that minimum_size > 0.

  Returns:
    A scalar integral tensor giving the next array size.
  """
  growth = tf.cast(_GROWTH_FACTOR, tf.float64)
  minimum_size = tf.cast(minimum_size, tf.float64)
  exponent = tf.math.ceil(tf.math.log(minimum_size) / tf.math.log(growth))
  return tf.cast(tf.math.ceil(tf.math.pow(growth, exponent)), tf.int64)


class Concatenator(keras.metrics.Metric):
  """Concatenate values across batches for exact large-scale metrics.

  This metric creates two local variables, `buffer` and `n_elements`, that are
  used to store concatenated values. Internally, `buffer` is used as storage
  for a dynamic array, which ensures that updates can be run in amortized
  constant time.
  """

  def __init__(self, dtype: tf.DType):
    """Constructs a concatenator metric.

    Args:
      dtype: The type of values to concatenate across batches.
    """
    super().__init__()
    self._buffer_dtype = dtype
    self._num_elements = tf.Variable(
        name="num_elements",
        trainable=False,
        initial_value=0,
        dtype=tf.int64,
    )
    self._buffer = tf.Variable(
        name="buffer",
        trainable=False,
        initial_value=[],
        # Allows assignment with values of different shapes.
        shape=tf.TensorShape(None),
        dtype=dtype,
    )

  def get_config(self) -> dict[str, Any]:
    return {"dtype": self._buffer_dtype}

  def result(self) -> tf.Tensor:
    """Returns all the concatenated values."""
    return self._buffer[: self._num_elements]

  @tf.function(reduce_retracing=True)
  def update_state(self, data: tf.Tensor):
    """Builds the concatenation graph.

    Args:
      data: A 1D tensor with the expected dtype.
    """
    new_num_elements = self._num_elements + tf.size(data, out_type=tf.int64)
    current_buffer_size = tf.size(self._buffer, out_type=tf.int64)

    def resize_buffer() -> tf.Operation:
      extra_buffer = tf.zeros(
          [_next_array_size(new_num_elements) - current_buffer_size],
          dtype=self._buffer.dtype,
      )
      new_buffer = tf.concat([self._buffer, extra_buffer], axis=0)
      return self._buffer.assign(new_buffer, read_value=False)

    must_resize = tf.greater(new_num_elements, current_buffer_size)
    maybe_resize = tf.cond(must_resize, resize_buffer, tf.no_op)
    with tf.control_dependencies([maybe_resize]):
      op = self._buffer[self._num_elements : new_num_elements].assign(data)

    with tf.control_dependencies([op]):
      self._num_elements.assign(new_num_elements, read_value=False)

  def reset_state(self):
    self._num_elements.assign(0)

  def merge_state(self, metrics: Iterable[keras.metrics.Metric]):
    elements = [self.result()]
    for metric in metrics:
      elements.append(metric.result())
    elements = tf.concat(elements, axis=0)
    n_elements = tf.size(elements, out_type=tf.int64)
    self._num_elements.assign(n_elements)
    self._buffer.assign(elements)
