from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.base import Block


def set_mapper_from_config(mapper_name, mapper_config):
    if mapper_name is None:
        return None
    name2mapper = {
        "LatentFactor": LatentFactorMapper,
    }
    if 'params' in mapper_config:
        return name2mapper[mapper_name](**mapper_config['params'])
    else:
        return name2mapper[mapper_name]()


def build_mappers(mapper_list):
    mapper_configs = [(k, v) for mapper in mapper_list for k, v in mapper.items()]
    mappers = [
        set_mapper_from_config(mapper[0], mapper[1])
        for mapper in mapper_configs
    ]
    return mappers


class LatentFactorMapper(Block):
    """
    latent factor mapper for cateory datas
    """

    def __init__(self,
                 id_num=None,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.id_num = id_num
        self.embedding_dim = embedding_dim

    def get_state(self):
        state = super().get_state()
        state.update({
            'id_num': self.id_num,
            'embedding_dim': self.embedding_dim})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.id_num = state['id_num']
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        input_node = list(inputs.values())[0]
        # TODO: better name for hp and better default for id_num
        id_num = self.id_num or hp.Choice('id_num', [10000], default=10000)
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16], default=8)
        output_node = tf.keras.layers.Embedding(id_num, embedding_dim)(input_node)
        return output_node
