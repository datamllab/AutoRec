from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Bias(Layer):
    """A keras layer with only bias term. 
      (can be understood as MLP layer with zero W matrix)
      
    # Attributes:
        units (int). The units of all layer in the Bias layer.
    """
    def __init__(self, units=32):
          super(Bias, self).__init__()

          bias_init = tf.zeros_initializer()
          self.bias = tf.Variable(initial_value=bias_init(shape=(units,),
                                                    dtype='float32'),
                               trainable=True)

    def call(self, inputs):
          return inputs + self.bias