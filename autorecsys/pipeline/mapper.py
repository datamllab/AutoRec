from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
import tensorflow as tf


def set_mapper_from_config(mapper_name, mapper_config):
    if mapper_name is None:
        return None
    name2mapper = {
        "LatentFactor": LatentFactorMapper,
    }
    return name2mapper[mapper_name](mapper_config)


def build_mappers(mapper_list):
    mapper_configs = [(k, v) for mapper in mapper_list for k, v in mapper.items()]
    mappers = [
        set_mapper_from_config(mapper[0], mapper[1])
        for mapper in mapper_configs
    ]
    return mappers


class BaseMapper(tf.keras.Model, metaclass=ABCMeta):
    def __init__(self, config, **kwarg):
        super(BaseMapper, self).__init__()
        self.config = config

    @abstractmethod
    def call(self, x):
        """call model."""
        raise NotImplementedError


class LatentFactorMapper(BaseMapper):
    """
    latent factor mapper for cateory datas
    """

    def __init__(self, config):
        super(LatentFactorMapper, self).__init__(config)

        if isinstance(self.config["params"]["id_num"], int):
            self.id_num = self.config["params"]["id_num"]
        elif isinstance(self.config["params"]["id_num"], list) and len(self.config["params"]["id_num"]) == 1:
            self.id_num = self.config["params"]["id_num"][0]
        else:
            raise ValueError("Id_num should be an integer or a list with length 1.")

        if isinstance(self.config["params"]["embedding_dim"], int):
            self.embedding_dim = self.config["params"]["embedding_dim"]
        elif isinstance(self.config["params"]["embedding_dim"], list) and len(self.config["params"]["embedding_dim"]) == 1:
            self.embedding_dim = self.config["params"]["embedding_dim"][0]
        else:
            raise ValueError("Embedding_dim should be an integer or a list with length 1.")

        self.user_embedding = tf.keras.layers.Embedding(self.id_num, self.embedding_dim)

    def call(self, x):
        # TODO: better implementation for dict inputs
        print()
        x = self.user_embedding(list(x.values())[0])
        return x
