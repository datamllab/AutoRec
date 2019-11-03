from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest
import unittest

import numpy as np
import tensorflow as tf
from autorecsys.pipeline.optimizer import (
    set_optimizer_from_config,
    build_optimizers,
    RatingPredictionOptimizer
)

logger = logging.getLogger(__name__)


class TestOptimizers(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_optimizer.ini").write("# testdata")

    def setUp(self):
        super(TestOptimizers, self).setUp()

        np.random.seed(123)
        tf.random.set_seed(4321)

        self.default_optimizer_name = "RatingPrediction"
        self.default_optimizer_config = {
            'params': {
                'latent_factor_dim': [10],
            }
        }
        self.optimizer_list = [{self.default_optimizer_name: self.default_optimizer_config} for _ in range(3)]

    def _assert_default_optimizer(self, optimizer):
        assert type(optimizer) == RatingPredictionOptimizer

    def test_set_optimizer_from_config(self):
        optimizer = set_optimizer_from_config(self.default_optimizer_name, self.default_optimizer_config)
        self._assert_default_optimizer(optimizer)

    def test_build_optimizers(self):
        optimizers = build_optimizers(self.optimizer_list)
        for idx in range(len(self.optimizer_list)):
            self._assert_default_optimizer(optimizers[idx])

    def test_RatingPredictionOptimizer(self):
        rp_optimizer = set_optimizer_from_config(self.default_optimizer_name, self.default_optimizer_config)
        # out_feat_dict = rp_optimizer(self.feat_dict)
        # assert