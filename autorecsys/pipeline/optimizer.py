from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.base import Block


class RatingPredictionOptimizer(Block):
    """Module for rating prediction task.
    # Attributes:
        None.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp, inputs=None):
        input_node = tf.concat(inputs, axis=1)
        output_node = tf.keras.layers.Dense(1)(input_node)
        output_node = tf.reshape(output_node, [-1])
        return output_node

    @property
    def metric(self):
        return tf.keras.metrics.MeanSquaredError(name='mse')

    @property
    def loss(self):
        return tf.keras.losses.MeanSquaredError(name='mse')



class PointWiseOptimizer(Block):
    """Module for click through rate prediction.
    # Attributes:
        None.
    """
    def build(self, hp, inputs=None):
        input_node = tf.concat(inputs, axis=1)
        output_node = tf.keras.layers.Dense(1, activation='sigmoid')(input_node)
        output_node = tf.reshape(output_node, [-1, 1])
        return output_node

    @property
    def metric(self):
        return tf.keras.metrics.BinaryCrossentropy(name='BinaryCrossentropy')
        # return tf.keras.metrics.AUC(name='AUC')

    @property
    def loss(self):
        return tf.keras.losses.BinaryCrossentropy(name='BinaryCrossentropy')
