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

from autorecsys.pipeline.preprocessor import Movielens1MCTRPreprocessor, Movielens1MPreprocessor


logger = logging.getLogger(__name__)


class TestPreprocessors(unittest.TestCase):
    # @pytest.fixture(autouse=True)
    # def initdir(self, tmpdir):
    #     tmpdir.chdir()  # change to pytest-provided temporary directory
    #     tmpdir.join("test_preprocessor.ini").write("# testdata")

    def setUp(self):
        super(TestPreprocessors, self).setUp()

        self.dataset_path = "../datasets/ml-1m/ratings.dat"
        column_names = ["user_id", "item_id", "rating"]
        tabular_data = np.array([
            [1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 4, 1],
            [2, 1, 1], [2, 2, 1], [2, 3, 1],
            [3, 1, 1], [3, 2, 1],
            [4, 1, 1]
        ])
        self.input_df = pd.DataFrame(tabular_data, columns=column_names)
        self.num_neg = 1

    def test_negative_sampling(self):
        # Test the following cases:
        #   1) Insufficient negative sampling candidates (user explored most items)
        #   2) Sufficient negative sampling candidates
        sol_num_pos = 10  # Arrange
        sol_num_neg = 4
        sol_type = pd.DataFrame
        mock_preprocessor = Movielens1MCTRPreprocessor("../datasets/ml-1m/ratings.dat")

        ans = mock_preprocessor._negative_sampling(self.input_df, self.num_neg)  # Act

        input(ans)

        ans_num_pos, ans_num_neg = ans["rating"].value_counts()[1], ans["rating"].value_counts()[0]  # Assert
        assert (ans_num_pos == sol_num_pos) & (ans_num_neg == sol_num_neg) & (type(ans) == sol_type)



    def test_Movielens1MPreprocessor(self):
        ml_1m = Movielens1MPreprocessor("../datasets/ml-1m/ratings.dat")
        ml_1m.preprocessing(test_size=0.2, random_state=1314)
        train_X, train_y, val_X, val_y = ml_1m.train_X, ml_1m.train_y, ml_1m.val_X, ml_1m.val_y
        print(train_X.shape)
        print(train_y.shape)
        print(val_X.shape)
        print(val_y.shape)
        print(train_X[:10])
        print(train_y[:10])
        print(type(train_X[:20][0][0]))
        print(type(train_y[:20][0]))


    def test_Movielens1MCTRPreprocessor(self):
        ml_1m = Movielens1MCTRPreprocessor("../datasets/ml-1m/ratings.dat")
        ml_1m.preprocessing(test_size=0.2, num_neg=10, random_state=1314)
        train_X, train_y, val_X, val_y = ml_1m.train_X, ml_1m.train_y, ml_1m.val_X, ml_1m.val_y
        print(train_X.shape)
        print(train_y.shape)
        print(val_X.shape)
        print(val_y.shape)
        print(train_X[:20])
        print(train_y[:20])
        print(type(train_X[:20][0][0]))
        print( type( train_y[:20][0] ) )
