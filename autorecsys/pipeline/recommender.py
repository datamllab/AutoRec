from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from autorecsys.utils import *
from autorecsys.pipeline.mapper import build_mappers
from autorecsys.pipeline.interactor import build_interactors
from autorecsys.pipeline.optimizer import build_optimizers


# class HyperRecommender(tf.keras.tuner.Hypermodel):
#
#     def build(self, hp):
#         a = hp.Boolean(name="bool", default=True)
#         hp.values['bool']
#         config = self.convert(hp)
#         return Recommender(config)


class Recommender(tf.keras.Model):
    def __init__(self, config):
        super(Recommender, self).__init__()

        self.config = load_config(config)
        self._build()

    def _build(self):

        self.mappers = build_mappers(self.config["Mapper"])
        self.interactors = build_interactors(self.config["Interactor"])
        self.optimizers = build_optimizers(self.config["Optimizer"])

    def call(self, feat_dict):

        # mapping
        mapper_output_dict = {}
        for mapper in self.mappers:
            mapper_output_dict[mapper.config["output"]] = mapper({k: feat_dict[k] for k in mapper.config["input"]})

        # interacting
        interactor_output_dict = {}
        for interactor in self.interactors:
            interactor_output_dict[interactor.config["output"]] = interactor(
                {k: mapper_output_dict[k] for k in interactor.config["input"]}
            )

        # predicting
        y_pred = {}
        for optimizer in self.optimizers:
            y_pred[optimizer.config["output"]] = optimizer(
                {k: interactor_output_dict[k] for k in optimizer.config["input"]}
            )

        return y_pred
