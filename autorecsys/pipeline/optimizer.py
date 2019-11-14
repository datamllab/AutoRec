from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.base import Block


def set_optimizer_from_config(optimizer_name, optimizer_config):
    if optimizer_name is None:
        return None
    name2optimizer = {
        "RatingPrediction": RatingPredictionOptimizer,
    }
    if 'params' in optimizer_config:
        return name2optimizer[optimizer_name](**optimizer_config['params'])
    else:
        return name2optimizer[optimizer_name]()


def build_optimizers(optimizer_list):
    optimizer_configs = [(k, v) for optimizer in optimizer_list for k, v in optimizer.items()]
    optimizers = [
        set_optimizer_from_config(optimizer[0], optimizer[1])
        for optimizer in optimizer_configs
    ]
    return optimizers


class RatingPredictionOptimizer(Block):
    """
    latent factor optimizer for cateory datas
    """

    def build(self, hp, inputs=None):
        input_node = inputs
        output_node = tf.concat([v for _, v in input_node.items()], axis=1)
        output_node = tf.keras.layers.Dense(1)(output_node)
        output_node = tf.reshape(output_node, [-1])
        return output_node
