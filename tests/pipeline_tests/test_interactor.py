from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import pytest
import unittest

import numpy as np
import tensorflow as tf
from autorecsys.pipeline.interactor import (
    ConcatenateInteraction, 
    ElementwiseAddInteraction,
    MLPInteraction,
)
from autorecsys.searcher.core import hyperparameters as hp_module

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU


logger = logging.getLogger(__name__)


class TestInteractors(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_interactor.ini").write("# testdata")

    def setUp(self):
        super(TestInteractors, self).setUp()
        self.inputs = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]


    def test_MLPInteraction(self):
        # TODO: Anthony
        pass

    def test_concatenate(self):
        """
        Test class ConcatenateInteraction in interactor.py
        """
        sol = tf.constant([1, 2, 3, 4, 5, 6])  # Arrange
        hp = hp_module.HyperParameters()
        interactor = ConcatenateInteraction()
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol))  # Assert


    def test_elementwise_add(self):
        """
        Test class ElementwiseAddInteraction in interactor.py
        """
        sol = tf.constant([5, 7, 9])  # Arrange
        hp = hp_module.HyperParameters()
        interactor = ElementwiseAddInteraction()
        ans = interactor.build(hp, self.inputs)  # Act
        assert all(tf.equal(ans, sol))  # Assert
