from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.base import Block


class LatentFactorMapper(Block):
    """
    latent factor mapper for single categorical feature
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
        id_num = self.id_num or hp.Choice('id_num', [10000], default=10000)
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16], default=8)
        output_node = tf.keras.layers.Embedding(id_num, embedding_dim)(input_node[0][:, self.feat_column_id])
        return output_node


class DenseFeatureMapper(Block):
    """
    dense feature mapper for dense type features
    """

    def __init__(self,
                 num_of_fields=None,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = ['num_of_fields']
        self.tunable_candidates = ['embedding_dim']
        self.num_of_fields = num_of_fields
        self.embedding_dim = embedding_dim
        self._check_fixed()
        self._hyperparameters = self._get_hyperparameters()

    def get_state(self):
        state = super().get_state()
        state.update({
            'num_of_fields': self.num_of_fields,
            'embedding_dim': self.embedding_dim})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.num_of_fields = state['num_of_fields']
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        input_node = inputs
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16], default=8)
        output_node = tf.stack(
            [
                tf.tensordot(input_node[0][:, feat_id], tf.keras.layers.Embedding(1, embedding_dim)(0), axes=0)
                for feat_id in range(self.num_of_fields)
            ],
            axis=1
        )
        return output_node


class SparseFeatureMapper(Block):
    """
    sparse feature mapper for categorical type features
    """

    def __init__(self,
                 num_of_fields=None,
                 hash_size=None,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = ['num_of_fields', 'hash_size']
        self.tunable_candidates = ['embedding_dim']
        self.num_of_fields = num_of_fields
        self.hash_size = hash_size
        self.embedding_dim = embedding_dim
        self._check_fixed()
        self._hyperparameters = self._get_hyperparameters()

    def get_state(self):
        state = super().get_state()
        state.update({
            'num_of_fields': self.num_of_fields,
            'hash_size': self.hash_size,
            'embedding_dim': self.embedding_dim})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.num_of_fields = state['num_of_fields']
        self.hash_size = state['hash_size']
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        input_node = inputs
        # TODO: modify default hash_size, current version is wrong when category of a feature is more
        # than 10000
        hash_size = self.hash_size or [hp.Choice('hash_size', [10000], default=10000)
                                       for _ in range(self.num_of_fields)]
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16], default=8)
        output_node = tf.stack(
            [
                tf.keras.layers.Embedding(hash_size[feat_id], embedding_dim)(input_node[0][:, feat_id])
                for feat_id in range(self.num_of_fields)
            ],
            axis=1
        )
        return output_node
