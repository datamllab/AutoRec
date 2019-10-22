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

        if not isinstance(self.config["input"], list):
            raise ValueError("Inputs of InnerProductInteraction should be a list.")
        elif len(self.config["input"]) != 2:
            raise ValueError("Inputs of InnerProductInteraction should be a list of length 2.")

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

        if isinstance(self.config["params"]["num_layers"], int):
            self.num_layers = self.config["params"]["num_layers"]
        elif isinstance(self.config["params"]["num_layers"], list) and len(self.config["params"]["num_layers"]) == 1:
            self.num_layers = self.config["params"]["num_layers"][0]
        else:
            raise ValueError("num_layers should be an integer or a list with length 1.")
        self.units = self.config["params"]["units"]

        if len(self.units) == 1 and self.num_layers > 1:
            self.units = self.units * self.num_layers

        if len(self.units) != self.num_layers:
            raise ValueError("Units should be a list of length 1 or the same number as num_layers.")

        for unit in self.units:
            self.dense_layers.append(tf.keras.layers.Dense(unit))

    def call(self, embeds):
        x = tf.concat([v for _, v in embeds.items()], axis=1)
        for layer in self.dense_layers:
            x = layer(x)
        return x
