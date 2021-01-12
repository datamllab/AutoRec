from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.base import Block


class RatingPredictionOptimizer(Block):
    """ For the rating prediction task, this module employs the default 'linear' activation function and the 'mse' (mean
    square error) loss and metric for training and evaluation.

    # Note
        This module takes a list of single tensor batch as input. When the input is a list of multiple tensor batches,
            they are concatenated into a single single tensor batch.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp, inputs=None):
        """ Build the optimization layer.

        # Arguments
            hp (HyperParameters): Specifies the search space and default value for the block's hyperparameters.
            inputs (Tensor): List of batch input tensors.

        # Returns
            The defined optimizer block.
        """
        input_node = tf.concat(inputs, axis=1)
        output_node = tf.keras.layers.Dense(1)(input_node)
        output_node = tf.reshape(output_node, [-1])
        return output_node

    @property
    def metric(self):
        """ Define the metric used for model evaluation.

        # Returns
            The defined metric object.
        """
        return tf.keras.metrics.MeanSquaredError(name='mse')

    @property
    def loss(self):
        """ Define the loss used for model training.

        # Returns
            The defined loss object.
        """
        return tf.keras.losses.MeanSquaredError(name='mse')


class CTRPredictionOptimizer(Block):
    """ For the CTR (click-through rate) prediction task, this module employs the 'sigmoid' activation function and
    the 'BinaryCrossentropy' loss and metric for training and evaluation.

    # Note
        This module takes a list of single tensor batch as input. When the input is a list of multiple tensor batches,
            they are concatenated into a single single tensor batch.
    """

    def build(self, hp, inputs=None):
        """ Build the optimization layer.

        # Arguments
            hp (HyperParameters): Specifies the search space and default value for the block's hyperparameters.
            inputs (Tensor): List of batch input tensors.

        # Returns
            The defined optimizer block.
        """
        input_node = tf.concat(inputs, axis=1)
        output_node = tf.keras.layers.Dense(1, activation='sigmoid')(input_node)
        output_node = tf.reshape(output_node, [-1, 1])
        return output_node

    @property
    def metric(self):
        """ Define the metric used for model evaluation.

        # Returns
            The defined metric object.
        """
        return tf.keras.metrics.BinaryCrossentropy(name='BinaryCrossentropy')

    @property
    def loss(self):
        """ Define the loss used for model training.

        # Returns
            The defined loss object.
        """
        return tf.keras.losses.BinaryCrossentropy(name='BinaryCrossentropy')
