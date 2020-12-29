from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import pytest
import unittest

import numpy as np
import tensorflow as tf
import pandas as pd
from autorecsys.pipeline.mapper import (
    DenseFeatureMapper,
    SparseFeatureMapper

)
from autorecsys.searcher.core import hyperparameters as hp_module

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU

logger = logging.getLogger(__name__)


class TestMappers(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_mapper.ini").write("# testdata")

    def setUp(self):
        super(TestMappers, self).setUp()
        self.col = 13
        self.batch = 2
        self.embed_dim = 8
        self.inputs = [tf.random.uniform([self.batch, self.col], dtype=tf.float32)]
        self.sparse_inputs = pd.DataFrame(np.random.rand(self.batch, self.col))
        # create pandas by series

    def test_DenseFeatureMapper(self):
        """
        Test class DenseFeatureMapper in mapper.py
        """

        hp = hp_module.HyperParameters()
        p = [self.col, self.embed_dim]
        interactor = DenseFeatureMapper(*p)
        ans = interactor.build(hp, self.inputs)  # Act
        assert isinstance(self.inputs, list), "input needs to be a list"
        assert ans.shape[0] == self.batch  # row
        assert ans.shape[1] == self.col  # col
        assert ans.shape[2] == self.embed_dim  # embed_dim

    def test_SparseFeatureMapper(self):
        """
        Test class SparseFeatureMapper in mapper.py
        """
        # add + 1 (?)
        hash_size = self.sparse_inputs.nunique().tolist()
        # hash_size = [x + 1 for x in hash_size]
        inputs = tf.convert_to_tensor(self.sparse_inputs.values, dtype=tf.int32)
        hp = hp_module.HyperParameters()
        p = [self.col, hash_size, self.embed_dim]
        interactor = SparseFeatureMapper(*p)
        ans = interactor.build(hp, [inputs])  # Act
        assert ans.shape[0] == self.batch  # Row
        assert ans.shape[1] == self.col  # Col
        assert ans.shape[2] == self.embed_dim  # embed_dim
