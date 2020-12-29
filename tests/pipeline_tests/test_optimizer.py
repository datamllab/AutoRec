from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import pytest
import unittest
import tensorflow as tf
from autorecsys.pipeline.optimizer import (
    RatingPredictionOptimizer,
)
from autorecsys.searcher.core import hyperparameters as hp_module

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU

logger = logging.getLogger(__name__)


class TestOptimizers(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_optimizer.ini").write("# testdata")

    def setUp(self):
        super(TestOptimizers, self).setUp()
        self.row = 2
        self.col = 13
        self.batch = 2
        self.inputs = [tf.random.uniform([self.batch, self.row*self.col], dtype=tf.float32)]
        # self.inputs = tf.constant([([1, 2], [3, 4]), ([5, 6], [7, 8])], dtype=tf.float32)

    def test_RatingPredictionOptimizer(self):
        """
        Test class RatingPredictionOptimizer in optimizer.py
        """
        # a = self.batch * self.row
        hp = hp_module.HyperParameters()
        interactor = RatingPredictionOptimizer()
        ans = interactor.build(hp, self.inputs)
        assert ans.shape == self.batch
