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
from tensorflow.python.util import nest


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU

logger = logging.getLogger(__name__)


class TestInteractors(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_interactor.ini").write("# testdata")

    def setUp(self):
        super(TestInteractors, self).setUp()
        self.inputs = [tf.constant([[1, 2, 3]], dtype='float32'),
                       tf.constant([[4, 5, 6]], dtype='float32')]

    def test_concatenate(self):
        """
        Test class ConcatenateInteraction in interactor.py
        """

        hp = hp_module.HyperParameters()
        interactor = ConcatenateInteraction()
        output = interactor.build(hp, self.inputs)  # Act

        # output
        assert len(nest.flatten(output)) == 1

        sol = tf.constant([[1, 2, 3, 4, 5, 6]], dtype=tf.float32)  # Arrange
        assert tf.reduce_all(tf.equal(output, sol))  # Assert

    def test_MLPInteraction(self):
        """
        Test class MLPInteraction in interactor.py
        """
        # intialize
        hp = hp_module.HyperParameters()
        p = {
            'units': 16,
            'num_layers': 2,
            'use_batchnorm': False,
            'dropout_rate': 0.25}  # units, num_layer, use_batchnorm, dropout
        interactor = MLPInteraction(**p)

        # test get_state()
        sol_get_state = {
            'name': 'mlp_interaction_1',
            'units': 16,
            'num_layers': 2,
            'use_batchnorm': False,
            'dropout_rate': 0.25}

        assert interactor.get_state() == sol_get_state

        # test set_state()
        p = {
            'units': 32,
            'num_layers': 1,
            'use_batchnorm': True,
            'dropout_rate': 0.0}
        sol_set_state = {
            'name': 'mlp_interaction_1',
            'units': 32,
            'num_layers': 1,
            'use_batchnorm': True,
            'dropout_rate': 0.0}
        interactor.set_state(p)
        ans_set_state = interactor.get_state()
        assert ans_set_state == sol_set_state

        # output shape
        output = interactor.build(hp, self.inputs)  # Act
        assert len(nest.flatten(output)) == 1

    def test_FMInteraction(self):
        """
        Test class FMInteraction in interactor.py
        """
        hp = hp_module.HyperParameters()
        p = {
            'embedding_dim': 8}  # embedding_dim
        interactor = FMInteraction(**p)

        # test get_state()
        sol_get_state = {
            'name': 'fm_interaction_1',
            'embedding_dim': 8}

        assert interactor.get_state() == sol_get_state

        # test set_state()
        p = {
            'embedding_dim': 16}
        sol_set_state = {
            'name': 'fm_interaction_1',
            'embedding_dim': 16}
        interactor.set_state(p)
        ans_set_state = interactor.get_state()
        assert ans_set_state == sol_set_state

        # output shape
        output = interactor.build(hp, self.inputs)  # Act
        assert len(nest.flatten(output)) == 1

    def test_CrossNetInteraction(self):
        """
        Test class CrossNetInteraction in interactor.py
        """
        hp = hp_module.HyperParameters()
        p = {
            'layer_num': 1}  # embedding_dim
        interactor = CrossNetInteraction(**p)

        # test get_state()
        sol_get_state = {
            'name': 'cross_net_interaction_1',
            'layer_num': 1}

        assert interactor.get_state() == sol_get_state

        # test set_state()
        p = {
            'layer_num': 2}
        sol_set_state = {
            'name': 'cross_net_interaction_1',
            'layer_num': 2}
        interactor.set_state(p)
        ans_set_state = interactor.get_state()
        assert ans_set_state == sol_set_state

        # output shape
        output = interactor.build(hp, self.inputs)  # Act
        assert len(nest.flatten(output)) == 1
