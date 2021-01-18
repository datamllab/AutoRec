from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Bias(Layer):
    """ This module builds a Keras layer of bias terms (e.g., MLP layer with zero weight matrix).

    # Arguments
        units (int): The units of all layer in the Bias layer.

    # Attributes
        bias (Tensor): The bias layer.
    """

    def __init__(self, units=32):
        super(Bias, self).__init__()
        bias_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=bias_init(shape=(units,), dtype='float32'), trainable=True)

    def call(self, inputs):
        """ Add the bias layer to the input tensor layer.

        # Arguments
            inputs (Tensor): List of batch input tensors.

        # Returns
            List of batch input tensors added with bias tensors.
        """
        return inputs + self.bias
