
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import pytest
import unittest

import numpy as np
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
        self.inputs = [tf.constant([1,2]), tf.constant([3,4])]


    def test_RatingPredictionOptimizer(self):
        # TODO: Anthony
        hp = hp_module.HyperParameters()
        opt = RatingPredictionOptimizer()
        check = opt.build(hp, self.inputs)
        
