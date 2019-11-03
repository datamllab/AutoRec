from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest
import unittest

import numpy as np
import tensorflow as tf
from autorecsys.pipeline.interactor import (
    set_interactor_from_config,
    build_interactors,
    InnerProductInteraction,
    MLPInteraction,
)

logger = logging.getLogger(__name__)


class TestInteractors(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_interactor.ini").write("# testdata")

    def setUp(self):
        super(TestInteractors, self).setUp()

        np.random.seed(123)
        tf.random.set_seed(4321)

        self.default_interactor_name = "MLP"
        self.default_interactor_config = {
            'params': {
                'num_layers': [3],
                'units': [2,3,4]
            }
        }
        self.interactor_list = [{
                        self.default_interactor_name: self.default_interactor_config
                        } for _ in range(3)]

    def _assert_default_interactor(self, interactor):
        assert type(interactor) == MLPInteraction
        assert interactor.num_layers == self.default_interactor_config["params"]["num_layers"][0]
        np.testing.assert_array_almost_equal(
            self.default_interactor_config["params"]["units"],
            interactor.units,
        )

    def test_set_interactor_from_config(self):
        interactor = set_interactor_from_config(self.default_interactor_name, self.default_interactor_config)
        self._assert_default_interactor(interactor)

    def test_build_interactors(self):
        interactors = build_interactors(self.interactor_list)
        for idx in range(len(self.interactor_list)):
            self._assert_default_interactor(interactors[idx])

    def test_MLPInteraction(self):
        mlp_interactor= set_interactor_from_config(self.default_interactor_name, self.default_interactor_config)
        # out_feat_dict = mlp_interactor(self.feat_dict)
        # assert