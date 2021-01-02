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

from autorecsys.pipeline.preprocessor import BasePreprocessor, NetflixPrizePreprocessor, CriteoPreprocessor, AvazuPreprocessor, MovielensPreprocessor


logger = logging.getLogger(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__)) # directory of this test file so that datasets are imported no mattter where the code is run
dataset_directory = os.path.join(current_directory,'../../examples/example_datasets')

class DummyPreprocessor(BasePreprocessor):
    """ Dummy class for testing base functions """

    def __init__(self,
                data_df=None,
                 non_csv_path=None,
                 csv_path=None,
                 header=0, 
                 columns=None,
                 delimiter='\t',
                 filler=0.0,
                 dtype_dict=None,  # inferred in load_data()
                 ignored_columns=None,
                 target_column='rating',
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=0, # all categories are counted
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        if columns is None:
            columns = range(3)
        if dtype_dict is None:
            dtype_dict = {}
        if ignored_columns is None:
            ignored_columns = []
        if numerical_columns is None:
            numerical_columns = ['num_people']
        if categorical_columns is None:
            categorical_columns = ['user_id']

        super().__init__(non_csv_path=non_csv_path,
                         csv_path=csv_path,
                         header=header,
                         delimiter=delimiter,
                         filler=filler,
                         dtype_dict=dtype_dict,
                         columns=columns,
                         ignored_columns=ignored_columns,
                         target_column=target_column,
                         numerical_columns=numerical_columns,
                         categorical_columns=categorical_columns,
                         categorical_filter=categorical_filter,
                         fit_dictionary_path=fit_dictionary_path,
                         transform_path=transform_path,
                         test_percentage=test_percentage,
                         validate_percentage=validate_percentage,
                         train_path=train_path,
                         validate_path=validate_path,
                         test_path=test_path)
        self.data_df = data_df

    def preprocess(self):
        self.transform_categorical()
        self.transform_numerical()
        x = self.get_x()
        y = self.get_y()
        x_train, x_test, y_train, y_test = self.split_data(x, y, self.test_percentage)
        x_train, x_validate, y_train, y_validate = self.split_data(x_train, y_train, self.validate_percentage)
        return x_train, y_train, x_validate, y_validate, x_test, y_test


class TestPreprocessors(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_preprocessor.ini").write("# testdata")

    def setUp(self):
        super(TestPreprocessors, self).setUp()

        column_names = ["user_id", "num_people", "rating"]
        tabular_data = np.array([
            [1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 4, 1],
            [2, 1, 1], [2, 2, 1], [2, 3, 1],
            [3, 1, 1], [3, 2, 1],
            [4, 1, 1]
        ])
        self.input_size = 10
        self.input_df = pd.DataFrame(tabular_data, columns=column_names)
    
    def test_BasePreprocessor(self):
        base = DummyPreprocessor(data_df=self.input_df)
        base.transform_numerical()
        assert base.data_df.shape == (10, 3)
        base.transform_categorical()
        assert base.data_df.shape == (10, 3)
        assert base.get_hash_size() == [4]

        x_cols = base.get_x()
        x_sol = self.input_df.drop(["rating"], axis=1)
        pd.testing.assert_frame_equal(x_sol, x_cols)
        assert np.array_equal(base.get_x_numerical(x_cols), x_sol[['num_people']].values)
        assert np.array_equal(base.get_x_categorical(x_cols), x_sol[['user_id']].values)

        y_vals_sol = np.ones(self.input_size)
        assert all(base.get_y() == y_vals_sol)
        assert base.get_numerical_count() == 1
        assert base.get_categorical_count() == 1

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
