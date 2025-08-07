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
"""SGD with sparse momentum and weight decay updates."""

from typing import Any, Optional
import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable("SGDSp")
class SGDSp(tf.keras.optimizers.Optimizer):
  r"""Gradient descent optimizer with sparse momentum and weight decay.

  This is identical to `tf.keras.optimizers.SGD` without Nesterov acceleration,
  except that for sparse gradients such as those typically encountered for
  embedding variables, momentum and decoupled weight decay only operate on the
  non-zero gradient slices of the variables. This can dramatically speed-up
  training on models with large embedding variables.

  The weight decay update is proportional to the step's learning rate times
  `sparse_weight_decay`. Following Loshchilov and Hutter in
  https://openreview.net/forum?id=Bkg6RiCqY7, to facilitate hyperparameter
  optimization, `sparse_weight_decay` should be controlled independently from
  the magnitude of the learning rate, for example by setting:
  `sparse_weight_decay = weight_decay_hyperparameter / peak_learning_rate`.
  """

  def __init__(
      self,
      learning_rate: (
          tf.keras.optimizers.schedules.LearningRateSchedule | float
      ) = 0.01,
      sparse_weight_decay: float = 0.0,
      sparse_momentum: float = 0.0,
      clipnorm: Optional[float] = None,
      clipvalue: Optional[float] = None,
      global_clipnorm: Optional[float] = None,
      use_ema: bool = False,
      ema_momentum: float = 0.99,
      ema_overwrite_frequency: Optional[int] = None,
      jit_compile: bool = True,
      name: str = "SGDSp",
      **kwargs,
  ):
    super().__init__(
        name=name,
        weight_decay=None,
        clipnorm=clipnorm,
        clipvalue=clipvalue,
        global_clipnorm=global_clipnorm,
        use_ema=use_ema,
        ema_momentum=ema_momentum,
        ema_overwrite_frequency=ema_overwrite_frequency,
        jit_compile=jit_compile,
        **kwargs,
    )
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.sm = sparse_momentum
    self.swd = sparse_weight_decay
    if isinstance(sparse_momentum, (int, float)) and (
        sparse_momentum < 0 or sparse_momentum > 1
    ):
      raise ValueError("`sparse_momentum` must be between [0, 1].")

  def build(self, var_list: list[tf.Variable]):
    """Initialize optimizer variables.

    If `self.sparse_momentum` is positive, SGDSp optimizer creates one
    additional variable per model variable in `self.momentum_vars`,
    otherwise `self.momentum_vars` is an empty list.

    Args:
        var_list: list of model variables to build SGDSp variables on.
    """
    super().build(var_list)
    if hasattr(self, "_built") and self._built:
      return

    self.momentum_vars = []
    if self.sm <= 0:
      self._built = True
      return

    for var in var_list:
      self.momentum_vars.append(
          self.add_variable_from_reference(
              model_variable=var, variable_name="m"
          )
      )
    self._built = True

  def _update_sparse_gradient(
      self,
      gradient: tf.IndexedSlices,
      variable: tf.Variable,
      lr: tf.Tensor,
      momentum_var: tf.Tensor,
      momentum: tf.Tensor,
  ):
    ix = gradient.indices
    gv = gradient.values
    m_updated_values = -gv * lr + tf.gather(momentum_var, ix) * momentum
    m_updated_is = tf.IndexedSlices(m_updated_values, ix)
    momentum_var.scatter_update(m_updated_is)
    variable.scatter_add(m_updated_is)

  def update_step(
      self, gradient: tf.Tensor | tf.IndexedSlices, variable: tf.Variable
  ):
    """Update step given gradient and the associated model variable."""
    lr = tf.cast(self.learning_rate, variable.dtype)
    wd = tf.cast(self.swd, variable.dtype)
    var_key = self._var_key(variable)
    momentum, momentum_var = None, None
    if self.momentum_vars:
      momentum = tf.cast(self.sm, variable.dtype)
      momentum_var = self.momentum_vars[self._index_dict[var_key]]

    if isinstance(gradient, tf.IndexedSlices):  # Sparse gradient.
      if self.swd > 0:  # Decay weights.
        variable.scatter_mul(tf.IndexedSlices(1.0 - wd * lr, gradient.indices))

      if momentum is None:
        add_value = tf.IndexedSlices(-gradient.values * lr, gradient.indices)
        variable.scatter_add(add_value)
      else:
        self._update_sparse_gradient(
            gradient, variable, lr, momentum_var, momentum
        )
    else:  # Dense gradient.
      if self.swd > 0:  # Decay weights.
        # tf.Variable has no `assign_mul` method.
        variable.assign_sub(variable * wd * lr)

      if momentum is None:
        variable.assign_add(-gradient * lr)
      else:
        momentum_var.assign(-gradient * lr + momentum_var * momentum)
        variable.assign_add(momentum_var)

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    del config["weight_decay"]

    config.update({
        "learning_rate": self._serialize_hyperparameter(self._learning_rate),
        "sparse_momentum": self.sm,
        "sparse_weight_decay": self.swd,
    })
    return config
