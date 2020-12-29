from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import pytest
import unittest

import tensorflow as tf
from autorecsys.pipeline.interactor import (
    ConcatenateInteraction,
    MLPInteraction,
    FMInteraction,
    CrossNetInteraction
)
from autorecsys.searcher.core import hyperparameters as hp_module
from .pipeline_tests_utils import layer_test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU

logger = logging.getLogger(__name__)

# tf.random.set_seed(1)


class TestInteractors(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_interactor.ini").write("# testdata")

    def setUp(self):
        super(TestInteractors, self).setUp()
        # self.inputs = tf.constant([([1, 2], [3, 4]), ([5, 6], [7, 8])], dtype=tf.float32)
        self.row = 2
        self.col = 13
        self.batch = 2
        self.inputs = tf.random.uniform([self.batch, self.row, self.col], dtype=tf.float32)

    def test_concatenate(self):
        """
        Test class ConcatenateInteraction in interactor.py
        """
        sol = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.float32)  # Arrange
        hp = hp_module.HyperParameters()
        interactor = ConcatenateInteraction()
        ans = interactor.build(hp, self.inputs)  # Act
        assert tf.reduce_all(tf.equal(ans, sol))  # Assert

    # def test_elementwise_add(self):
    #     """
    #     Test class ElementwiseAddInteraction in interactor.py
    #     """
    #     sol = tf.constant([[6, 8], [10, 12]])  # p
    #     hp = hp_module.HyperParameters()
    #     interactor = ElementwiseInteraction(elementwise_type="sum")
    #     ans = interactor.build(hp, self.inputs)  # Act
    #     assert tf.reduce_all((tf.equal(ans, sol)))  # Assert

    def test_MLPInteraction(self):
        """
        Test class ElementwiseAddInteraction in interactor.py
        """
        hp = hp_module.HyperParameters()
        p = [32, 2, False, .25]  # units, num_layer, use_batchnorm, dropout
        interactor = MLPInteraction(*p)
        ans = interactor.build(hp, self.inputs)  # Act
        layer_test(interactor, self.inputs, ans, p, name='MLP')

    def test_FMInteraction(self):
        """
        Test class FMInteraction in interactor.py
        """
        hp = hp_module.HyperParameters()
        p = [8]  # embed_dim
        interactor = FMInteraction(*p)
        ans = interactor.build(hp, self.inputs)  # Act
        layer_test(interactor, self.inputs, ans, p, name='FM')

    def test_CrossNetInteraction(self):
        """
        Test class CrossNetInteraction in interactor.py
        """
        hp = hp_module.HyperParameters()
        p = [1]  # num_layer
        interactor = CrossNetInteraction(*p)
        ans = interactor.build(hp, self.inputs)  # Act
        layer_test(interactor, self.inputs, ans, p, name='CrossNet')
