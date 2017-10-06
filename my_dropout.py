import tensorflow as tf
import collections
import contextlib
import hashlib
import math
import numbers

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import RNNCell as RNNCell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

def _enumerated_map_structure(map_fn, *args, **kwargs):
  ix = [0]
  def enumerated_fn(*inner_args, **inner_kwargs):
    r = map_fn(ix[0], *inner_args, **inner_kwargs)
    ix[0] += 1
    return r
  return nest.map_structure(enumerated_fn, *args, **kwargs)

class MyDropoutWrapper(RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               state_keep_prob=1.0, variational_recurrent=False,
               input_size=None, dtype=None, seed=None):
    """Create a cell with added input, state, and/or output dropout.
    If `variational_recurrent` is set to `True` (**NOT** the default behavior),
    then the the same dropout mask is applied at every step, as described in:
    Y. Gal, Z Ghahramani.  "A Theoretically Grounded Application of Dropout in
    Recurrent Neural Networks".  https://arxiv.org/abs/1512.05287
    Otherwise a different dropout mask is applied at every time step.
    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is constant and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
      state_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
        State dropout is performed on the *output* states of the cell.
      variational_recurrent: Python bool.  If `True`, then the same
        dropout pattern is applied across all time steps per run call.
        If this parameter is set, `input_size` **must** be provided.
      input_size: (optional) (possibly nested tuple of) `TensorShape` objects
        containing the depth(s) of the input tensors expected to be passed in to
        the `DropoutWrapper`.  Required and used **iff**
         `variational_recurrent = True` and `input_keep_prob < 1`.
      dtype: (optional) The `dtype` of the input, state, and output tensors.
        Required and used **iff** `variational_recurrent = True`.
      seed: (optional) integer, the randomness seed.
    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if any of the keep_probs are not between 0 and 1.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")
    with ops.name_scope("DropoutWrapperInit"):
      def tensor_and_const_value(v):
        tensor_value = ops.convert_to_tensor(v)
        const_value = tensor_util.constant_value(tensor_value)
        return (tensor_value, const_value)
      for prob, attr in [(input_keep_prob, "input_keep_prob"),
                         (state_keep_prob, "state_keep_prob"),
                         (output_keep_prob, "output_keep_prob")]:
        tensor_prob, const_prob = tensor_and_const_value(prob)
        if const_prob is not None:
          if const_prob < 0 or const_prob > 1:
            raise ValueError("Parameter %s must be between 0 and 1: %d"
                             % (attr, const_prob))
          setattr(self, "_%s" % attr, float(const_prob))
        else:
          setattr(self, "_%s" % attr, tensor_prob)

    # Set cell, variational_recurrent, seed before running the code below
    self._cell = cell
    self._variational_recurrent = variational_recurrent
    self._seed = seed

    self._recurrent_input_noise = None
    self._recurrent_state_noise = None
    self._recurrent_output_noise = None

    if variational_recurrent:
      if dtype is None:
        raise ValueError(
            "When variational_recurrent=True, dtype must be provided")

      def convert_to_batch_shape(s):
        # Prepend a 1 for the batch dimension; for recurrent
        # variational dropout we use the same dropout mask for all
        # batch elements.
        return array_ops.concat(
            ([1], tensor_shape.TensorShape(s).as_list()), 0)

      def batch_noise(s, inner_seed):
        shape = convert_to_batch_shape(s)
        return random_ops.random_uniform(shape, seed=inner_seed, dtype=dtype)

      if (not isinstance(self._input_keep_prob, numbers.Real) or
          self._input_keep_prob < 1.0):
        if input_size is None:
          raise ValueError(
              "When variational_recurrent=True and input_keep_prob < 1.0 or "
              "is unknown, input_size must be provided")
        self._recurrent_input_noise = _enumerated_map_structure(
            lambda i, s: batch_noise(s, inner_seed=self._gen_seed("input", i)),
            input_size)
      self._recurrent_state_noise = _enumerated_map_structure(
          lambda i, s: batch_noise(s, inner_seed=self._gen_seed("state", i)),
          cell.state_size.h)
      self._recurrent_output_noise = _enumerated_map_structure(
          lambda i, s: batch_noise(s, inner_seed=self._gen_seed("output", i)),
          cell.output_size)

  def _gen_seed(self, salt_prefix, index):
    if self._seed is None:
      return None
    salt = "%s_%d" % (salt_prefix, index)
    string = (str(self._seed) + salt).encode("utf-8")
    return int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def _variational_recurrent_dropout_value(
      self, index, value, noise, keep_prob):
    """Performs dropout given the pre-calculated noise tensor."""
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob + noise

    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    ret = math_ops.div(value, keep_prob) * binary_tensor
    ret.set_shape(value.get_shape())
    return ret

  def _dropout(self, values, salt_prefix, recurrent_noise, keep_prob):
    """Decides whether to perform standard dropout or recurrent dropout."""
    if not self._variational_recurrent:
      def dropout(i, v):
        return nn_ops.dropout(
            v, keep_prob=keep_prob, seed=self._gen_seed(salt_prefix, i))
      return _enumerated_map_structure(dropout, values)
    else:
      def dropout(i, v, n):
        return self._variational_recurrent_dropout_value(i, v, n, keep_prob)
      return _enumerated_map_structure(dropout, values, recurrent_noise)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with the declared dropouts."""
    def _should_dropout(p):
      return (not isinstance(p, float)) or p < 1

    if _should_dropout(self._input_keep_prob):
      inputs = self._dropout(inputs, "input",
                             self._recurrent_input_noise,
                             self._input_keep_prob)
    output, new_state = self._cell(inputs, state, scope)
    if _should_dropout(self._state_keep_prob):
      c, h = new_state
      h = self._dropout(h, "state",
                        self._recurrent_state_noise,
                        self._state_keep_prob)
      new_state = tf.contrib.rnn.LSTMStateTuple(c, h)
    if _should_dropout(self._output_keep_prob):
      output = self._dropout(output, "output",
                             self._recurrent_output_noise,
                             self._output_keep_prob)
    return output, new_state