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

from autorecsys.pipeline.preprocessor import NetflixPrizePreprocessor, CriteoPreprocessor, AvazuPreprocessor, MovielensPreprocessor


logger = logging.getLogger(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__)) # directory of this test file so that datasets are imported no mattter where the code is run
dataset_directory = os.path.join(current_directory,'../../examples/example_datasets')


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

    def test_MovielensPreprocessor(self):
        movielens = MovielensPreprocessor(csv_path=os.path.join(dataset_directory,'movielens/ratings-10k.dat'))
        train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()
        print(train_X.shape)
        print(train_y.shape)
        print(val_X.shape)
        print(val_y.shape)
        print(test_X.shape)
        print(test_y.shape)
        print(train_X[:10])
        print(train_y[:10])
        assert movielens.data_df.shape == (10000, 3) # check shape to verify transform functions
        assert movielens.get_categorical_count() == 2
        assert movielens.get_numerical_count() == 0
        assert movielens.get_x().shape[0] == movielens.get_y().shape[0] # check x and y have same length
        assert len(movielens.get_hash_size()) == movielens.get_categorical_count()
        assert movielens.get_numerical_count() + movielens.get_categorical_count() == len(movielens.get_x().columns) # check numerical + categorical = total columns

    def test_CriteoPreprocessor(self):
        criteo = CriteoPreprocessor(csv_path=os.path.join(dataset_directory,'criteo/train-10k.txt'))
        train_X, train_y, val_X, val_y, test_X, test_y = criteo.preprocess()
        print(train_X.shape)
        print(train_y.shape)
        print(val_X.shape)
        print(val_y.shape)
        print(test_X.shape)
        print(test_y.shape)
        print(train_X[:10])
        print(train_y[:10])
        assert criteo.data_df.shape == (10000, 40)
        assert criteo.get_categorical_count() == 26
        assert criteo.get_numerical_count() == 13
        assert criteo.get_x().shape[0] == criteo.get_y().shape[0]
        assert len(criteo.get_hash_size()) == criteo.get_categorical_count()
        assert criteo.get_numerical_count() + criteo.get_categorical_count() == len(criteo.get_x().columns)

    def test_NetflixPreprocessor(self):
        netflix = NetflixPrizePreprocessor(non_csv_path=os.path.join(dataset_directory,'netflix/combined_data_1-10k.txt'), csv_path=os.path.join(dataset_directory,'netflix/combined_data_1-10k.csv'))
        train_X, train_y, val_X, val_y, test_X, test_y = netflix.preprocess()
        print(train_X.shape)
        print(train_y.shape)
        print(val_X.shape)
        print(val_y.shape)
        print(test_X.shape)
        print(test_y.shape)
        print(train_X[:10])
        print(train_y[:10])
        assert netflix.data_df.shape == (10000, 3)
        assert netflix.get_categorical_count() == 2
        assert netflix.get_numerical_count() == 0
        assert netflix.get_x().shape[0] == netflix.get_y().shape[0]
        assert len(netflix.get_hash_size()) == netflix.get_categorical_count()
        assert netflix.get_numerical_count() + netflix.get_categorical_count() == len(netflix.get_x().columns)

    def test_AvazuPreprocessor(self):
        avazu = AvazuPreprocessor(csv_path=os.path.join(dataset_directory,'avazu/train-10k'))
        train_X, train_y, val_X, val_y, test_X, test_y = avazu.preprocess()
        print(train_X.shape)
        print(train_y.shape)
        print(val_X.shape)
        print(val_y.shape)
        print(test_X.shape)
        print(test_y.shape)
        print(train_X[:10])
        print(train_y[:10])
        assert avazu.data_df.shape == (9999, 23)
        assert avazu.get_categorical_count() == 22
        assert avazu.get_numerical_count() == 0
        assert avazu.get_x().shape[0] == avazu.get_y().shape[0]
        assert len(avazu.get_hash_size()) == avazu.get_categorical_count()
        assert avazu.get_numerical_count() + avazu.get_categorical_count() == len(avazu.get_x().columns)


