from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from abc import ABCMeta, abstractmethod


def set_interactor_from_config(interactor_name, interactor_config):
    if interactor_name is None:
        return None
    name2interactor = {
        "MLP": MLPInteraction,
        "InnerProduct": InnerProductInteraction,
    }
    return name2interactor[interactor_name](interactor_config)


def build_interactors(interactor_list):
    interactor_configs = [(k, v) for interactor in interactor_list for k, v in interactor.items()]
    interactors = [
        set_interactor_from_config(interactor[0], interactor[1])
        for interactor in interactor_configs
    ]
    return interactors


class BaseInteraction(tf.keras.Model, metaclass=ABCMeta):
    def __init__(self, config):
        super(BaseInteraction, self).__init__(config)
        self.config = config

    @abstractmethod
    def call(self):
        raise NotImplementedError


class InnerProductInteraction(BaseInteraction):
    """
    latent factor interactor for category datas
    """

    def __init__(self, config):
        super(InnerProductInteraction, self).__init__(config)

    def call(self, embeds):
        x = embeds[self.config["input"][0]] * embeds[self.config["input"][1]]
        return x


class MLPInteraction(BaseInteraction):
    """
    latent factor interactor for cateory datas
    """

    def __init__(self, config):
        super(MLPInteraction, self).__init__(config)
        self.dense_layers = []
        for dim in self.config["params"]["layer_output_dim"]:
            self.dense_layers.append(tf.keras.layers.Dense(dim))

    def call(self, embeds):
        x = tf.concat([v for _, v in embeds.items()], axis=1)
        for layer in self.dense_layers:
            x = layer(x)
        return x
