from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.base import Block


class LatentFactorMapper(Block):
    """ This module maps the user (item) entity into embeddings (latent factors).

    # Note
        Data-wise, the similarity b/t class LatentFactorMapper and class SparseFeatureMapper is that both user (item)
            identifiers and indexed categorical data are sparse and devoid of numerical meaning.
        Functionally, the difference b/t class LatentFactorMapper and class SparseFeatureMapper is that they handle one
            sparse column (either user or item) and multiple sparse columns (categorical features), respectively.
        In terms of nomenclature, the difference b/t class LatentFactorMapper and class SparseFeatureMapper is to
            distinguish the host of features (user and item) from the features themselves.
        The use of the term "latent factor" can be traced back to early matrix factorization models for recommendation,
            which involve only user and item.
        Reference: https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

    # Arguments
        column_id (int): The index of the user (item) entity column.
        num_of_entities (int): The number of the user (item) entity.
        embedding_dim (int): The dimension of the embeddings (latent factors).

    # Attributes
        column_id (int): The index of the user (item) entity column.
        num_of_entities (int): The number of the user (item) entities.
        embedding_dim (int): The dimension of the embeddings (latent factors).
    """

    def __init__(self,
                 column_id=None,
                 num_of_entities=None,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.column_id = column_id
        self.num_of_entities = num_of_entities
        self.embedding_dim = embedding_dim

    def get_state(self):
        state = super().get_state()
        state.update({
            'column_id': self.column_id,
            'num_of_entities': self.num_of_entities,
            'embedding_dim': self.embedding_dim})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.column_id = state['column_id']
        self.num_of_entities = state['num_of_entities']
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        input_node = inputs
        num_of_entities = self.num_of_entities or hp.Choice('num_of_entities', [10000], default=10000)
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16, 32, 64, 128], default=32)
        output_node = tf.keras.layers.Embedding(num_of_entities, embedding_dim)(input_node[0][:, self.column_id])
        return output_node


class SparseFeatureMapper(Block):
    """ This module maps the categorical data of sparse feature columns into embeddings.

    # Arguments
        num_of_fields (int): The number of sparse feature columns (fields).
        hash_size (list): The numbers of categories used in each sparse feature column.
        embedding_dim (int): The dimension of the embeddings.

    # Attributes
        num_of_fields (int): The number of sparse feature columns (fields).
        hash_size (list): The list of numbers of categories used in each sparse feature column.
        embedding_dim (int): The dimension of the embeddings.
    """

    def __init__(self,
                 num_of_fields=None,
                 hash_size=None,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_of_fields = num_of_fields
        self.hash_size = hash_size
        self.embedding_dim = embedding_dim

    def get_state(self):
        """ Get information about the mapper layer, including name, level, and hyperparameters.

        # Returns
            Dictionary where key=attribute name and val=attribute value.
        """
        state = super().get_state()
        state.update({
            'num_of_fields': self.num_of_fields,
            'hash_size': self.hash_size,
            'embedding_dim': self.embedding_dim})
        return state

    def set_state(self, state):
        """ Set information about the mapper layer, including name, level, and hyperparameters.

        # Arguments
            state (dict): Map attribute names to attribute values.
        """
        super().set_state(state)
        self.num_of_fields = state['num_of_fields']
        self.hash_size = state['hash_size']
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        """ Build the mapper layer.

        Note:
            Attribute "hash_size" has search space [10000]. Default is 10000.
            Attribute "embedding_dim" has search space [8, 16]. Default is 8.

        # Arguments
            hp (HyperParameters): Specifies the search space and default value for the block's hyperparameters.
            inputs (Tensor): List of batch input tensors.

         # Returns
            The defined mapper block.
        """
        input_node = inputs
        # TODO: modify default hash_size, current version is wrong when category of a feature is more than 10000
        hash_size = self.hash_size or [hp.Choice('hash_size', [10000], default=10000)
                                       for _ in range(self.num_of_fields)]
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16], default=8)
        output_node = tf.stack(
            [
                tf.keras.layers.Embedding(hash_size[col_id], embedding_dim)(input_node[0][:, col_id])
                for col_id in range(self.num_of_fields)
            ],
            axis=1
        )
        return output_node


class DenseFeatureMapper(Block):
    """ This module maps the numerical data of dense feature columns into embeddings.

    # Arguments
        num_of_fields (int): The number of dense feature columns.
        embedding_dim (int): The dimension of the embeddings.

    # Attributes
        num_of_fields (int): The number of dense feature columns.
        embedding_dim (int): The dimension of the embeddings.
    """

    def __init__(self,
                 num_of_fields=None,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_of_fields = num_of_fields
        self.embedding_dim = embedding_dim

    def get_state(self):
        """ Get information about the mapper layer, including name, level, and hyperparameters.

        # Returns
            Dictionary where key=attribute name and val=attribute value.
        """
        state = super().get_state()
        state.update({
            'num_of_fields': self.num_of_fields,
            'embedding_dim': self.embedding_dim})
        return state

    def set_state(self, state):
        """ Set information about the mapper layer, including name, level, and hyperparameters.

        # Arguments
            state (dict): Map attribute names to attribute values.
        """
        super().set_state(state)
        self.num_of_fields = state['num_of_fields']
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        """ Build the mapper layer.

        Note:
            Attribute "embedding_dim" has search space [8, 16, 32]. Default is 8.

        # Arguments
            hp (HyperParameters): Specifies the search space and default value for the block's hyperparameters.
            inputs (Tensor): List of batch input tensors.

         # Returns
            The defined mapper block.
        """
        input_node = inputs
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [8, 16], default=8)
        output_node = tf.stack(
            [
                tf.tensordot(input_node[0][:, col_id], tf.keras.layers.Embedding(1, embedding_dim)(0), axes=0)
                for col_id in range(self.num_of_fields)
            ],
            axis=1
        )
        return output_node

