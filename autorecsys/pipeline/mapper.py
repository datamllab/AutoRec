from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.base import Block


class LatentFactorMapper(Block):
    """
    latent factor mapper for cateory datas
    """

    def __init__(self,
                 feat_column_id=None,
                 id_num=None,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = ['feat_column_id', 'id_num']
        self.tunable_candidates = ['embedding_dim']
        self.feat_column_id = feat_column_id
        self.id_num = id_num
        self.embedding_dim = embedding_dim
        self._check_fixed()
        self._hyperparameters = self._get_hyperparameters()

    def get_state(self):
        state = super().get_state()
        state.update({
            'feat_column_id': self.feat_column_id,
            'id_num': self.id_num,
            'embedding_dim': self.embedding_dim})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.feat_column_id = state['feat_column_id']
        self.id_num = state['id_num']
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        input_node = inputs
        # TODO: better name for hp and better default for id_num
        id_num = self.id_num or hp.Choice('id_num', [10000], default=10000)
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16], default=8)
        output_node = tf.keras.layers.Embedding(id_num, embedding_dim)(input_node[0][:, self.feat_column_id])
        return output_node
