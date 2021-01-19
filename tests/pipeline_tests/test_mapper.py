from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import pytest
import unittest

import numpy as np
import tensorflow as tf
import pandas as pd
from autorecsys.pipeline.mapper import (
    LatentFactorMapper,
    DenseFeatureMapper,
    SparseFeatureMapper
)
from autorecsys.searcher.core import hyperparameters as hp_module
from tensorflow.python.util import nest


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU

logger = logging.getLogger(__name__)


class TestMappers(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_mapper.ini").write("# testdata")

    def setUp(self):
        super(TestMappers, self).setUp()
        self.column_id = 1
        self.num_of_entities = 10
        self.input_shape = 13
        self.batch = 2
        self.embed_dim = 8
        self.dense_inputs = [tf.random.uniform([self.batch, self.input_shape], dtype=tf.float32)]
        self.sparse_inputs = pd.DataFrame(np.random.rand(self.batch, self.input_shape))

    def test_LatentFactorMapper(self):
        # test constructor and get_state
        p = {
            'column_id': 0,
            'num_of_entities': 3,
            'embedding_dim': 4}
        mapper = LatentFactorMapper(**p)
        sol_get_state = {
            'name': 'latent_factor_mapper_1',
            'column_id': 0,
            'num_of_entities': 3,
            'embedding_dim': 4}
        assert mapper.get_state() == sol_get_state

        # test set_state
        p = {
            'column_id': self.column_id,
            'num_of_entities': self.num_of_entities,
            'embedding_dim': self.embed_dim}
        sol_set_state = {
            'name': 'latent_factor_mapper_1',
            'column_id': self.column_id,
            'num_of_entities': self.num_of_entities,
            'embedding_dim': self.embed_dim}
        mapper.set_state(p)
        ans_set_state = mapper.get_state()
        assert ans_set_state == sol_set_state

        # test build
        hp = hp_module.HyperParameters()
        output = mapper.build(hp, self.dense_inputs)  # Act
        assert len(nest.flatten(output)) == 1
        assert output.shape == (self.batch, self.embed_dim)  # LatentFactorMapper does not have input shape dimension

    def test_DenseFeatureMapper(self):
        # test constructor and get_state
        p = {
            'num_of_fields': 10,
            'embedding_dim': 4}
        mapper = DenseFeatureMapper(**p)
        sol_get_state = {
            'name': 'dense_feature_mapper_1',
            'num_of_fields': 10,
            'embedding_dim': 4}
        assert mapper.get_state() == sol_get_state

        # test set_state
        p = {
            'num_of_fields': self.input_shape,
            'embedding_dim': self.embed_dim}
        sol_set_state = {
            'name': 'dense_feature_mapper_1',
            'num_of_fields': self.input_shape,
            'embedding_dim': self.embed_dim}
        mapper.set_state(p)
        ans_set_state = mapper.get_state()
        assert ans_set_state == sol_set_state

        # test build
        hp = hp_module.HyperParameters()
        output = mapper.build(hp, self.dense_inputs)  # Act
        assert len(nest.flatten(output)) == 1
        assert output.shape == (self.batch, self.input_shape, self.embed_dim)

    def test_SparseFeatureMapper(self):
        # test constructor and get_state
        p = {
            'num_of_fields': 10,
            'hash_size': [2, 4, 10],
            'embedding_dim': 4}
        mapper = SparseFeatureMapper(**p)
        sol_get_state = {
            'name': 'sparse_feature_mapper_1',
            'num_of_fields': 10,
            'hash_size': [2, 4, 10],
            'embedding_dim': 4}
        assert mapper.get_state() == sol_get_state

        # test set_state
        hash_size = self.sparse_inputs.nunique().tolist()
        p = {
            'num_of_fields': self.input_shape,
            'hash_size': hash_size,
            'embedding_dim': self.embed_dim}
        sol_set_state = {
            'name': 'sparse_feature_mapper_1',
            'num_of_fields': self.input_shape,
            'hash_size': hash_size,
            'embedding_dim': self.embed_dim}
        mapper.set_state(p)
        ans_set_state = mapper.get_state()
        assert ans_set_state == sol_set_state

        # test build
        hp = hp_module.HyperParameters()
        inputs = [tf.convert_to_tensor(self.sparse_inputs.values, dtype=tf.int32)]
        mapper = SparseFeatureMapper(**p)
        output = mapper.build(hp, inputs)  # Act
        assert len(nest.flatten(output)) == 1
        assert output.shape == (self.batch, self.input_shape, self.embed_dim)
