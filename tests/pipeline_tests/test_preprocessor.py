from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.utils import shuffle

import os
import random
import functools
import logging
import pytest
import unittest

import pandas as pd
import numpy as np
import tensorflow as tf

from autorecsys.pipeline.preprocessor import (
    negative_sampling
)

logger = logging.getLogger(__name__)


class TestPreprocessors(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_preprocessor.ini").write("# testdata")

    def setUp(self):
        super(TestPreprocessors, self).setUp()

        column_names = ["user_id", "item_id", "rating"]
        tabular_data = np.array([
            [1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 4, 1],
            [2, 1, 1], [2, 2, 1], [2, 3, 1],
            [3, 1, 1], [3, 2, 1],
            [4, 1, 1]
        ])
        self.input_df = pd.DataFrame(tabular_data, columns=column_names)
        self.num_neg = 1
        self.seed = 42

    def test_negative_sampling(self):
        # Test the following cases:
        #   1) Insufficient negative sampling candidates (user explored most items)
        #   2) Sufficient negative sampling candidates
        sol_num_pos = 10  # Arrange
        sol_num_neg = 4
        sol_type = pd.DataFrame

        ans = negative_sampling(self.input_df, self.num_neg, self.seed)  # Act

        ans_num_pos, ans_num_neg = ans["rating"].value_counts()[1], ans["rating"].value_counts()[0]  # Assert
        assert (ans_num_pos == sol_num_pos) & (ans_num_neg == sol_num_neg) & (type(ans) == sol_type)

