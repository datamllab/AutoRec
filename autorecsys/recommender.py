import tensorflow as tf

from autorecsys.utils import load_config
from autorecsys.mapper import build_mappers
from autorecsys.interaction import build_interactors
from autorecsys.optimizer import build_optimizers


class Recommender(tf.keras.Model):
    def __init__(self, config_filename):
        super(Recommender, self).__init__()

        self.config = load_config(config_filename)
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
