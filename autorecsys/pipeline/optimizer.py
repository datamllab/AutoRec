from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from abc import ABCMeta, abstractmethod


def set_optimizer_from_config(optimizer_name, optimizer_config):
    if optimizer_name is None:
        return None
    name2optimizer = {
        "RatingPrediction": RatingPredictionOptimizer,
    }
    return name2optimizer[optimizer_name](optimizer_config)


def build_optimizers(optimizer_list):
    optimizer_configs = [(k, v) for optimizer in optimizer_list for k, v in optimizer.items()]
    optimizers = [
        set_optimizer_from_config(optimizer[0], optimizer[1])
        for optimizer in optimizer_configs
    ]
    return optimizers


class BaseOptimizer(tf.keras.Model, metaclass=ABCMeta):
    def __init__(self, config, **kwargs):
        super(BaseOptimizer, self).__init__(config)
        self.config = config

    @abstractmethod
    def call(self, x):
        """call model."""


class RatingPredictionOptimizer(BaseOptimizer):
    """
    latent factor optimizer for cateory datas
    """

    def __init__(self, config):
        super(RatingPredictionOptimizer, self).__init__(config)
        self.dense_layer = tf.keras.layers.Dense(1)

    def call(self, embeds):
        x = tf.concat([v for _, v in embeds.items()], axis=1)
        x = self.dense_layer(x)
        x = tf.reshape(x, [-1])
        return x
