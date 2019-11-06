from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest
import unittest

import numpy as np
import tensorflow as tf
from autorecsys.pipeline.mapper import (
    set_mapper_from_config,
    build_mappers,
    LatentFactorMapper
)

logger = logging.getLogger(__name__)


class TestMappers(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_mapper.ini").write("# testdata")

    def setUp(self):
        super(TestMappers, self).setUp()

        np.random.seed(123)
        tf.random.set_seed(4321)

        self.default_mapper_name = "LatentFactor"
        self.default_mapper_config = {
            'params': {
                'embedding_dim': [10],
                'id_num': 3
            }
        }
        self.mapper_list = [{self.default_mapper_name: self.default_mapper_config} for _ in range(3)]

    def _assert_default_mapper(self, mapper):
        assert type(mapper) == LatentFactorMapper
        assert mapper.user_embedding.input_dim == self.default_mapper_config["params"]["id_num"]
        assert mapper.user_embedding.output_dim == self.default_mapper_config["params"]["embedding_dim"][0]

    def test_set_mapper_from_config(self):
        mapper = set_mapper_from_config(self.default_mapper_name, self.default_mapper_config)
        self._assert_default_mapper(mapper)

    def test_build_mappers(self):
        mappers = build_mappers(self.mapper_list)
        for idx in range(len(self.mapper_list)):
            self._assert_default_mapper(mappers[idx])

    def test_LatentFactorMapper(self):
        lf_mapper = set_mapper_from_config(self.default_mapper_name, self.default_mapper_config)
        # TODO: Anthony (similar to the the interactor test)
        # out_feat_dict = lf_mapper(self.feat_dict)
        # assert
        