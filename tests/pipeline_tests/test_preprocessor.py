from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.utils import shuffle

import os
import random
import functools
import logging
import pytest
import unittest

import math
import pandas as pd
import numpy as np
import tensorflow as tf

from autorecsys.pipeline.preprocessor import BasePreprocessor, NetflixPrizePreprocessor, CriteoPreprocessor, AvazuPreprocessor, MovielensPreprocessor


logger = logging.getLogger(__name__)

# directory of this test file so that datasets are imported no mattter where the code is run
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_directory = os.path.join(
    current_directory, '../../examples/example_datasets')


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
                 categorical_filter=0,  # all categories are counted
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
        return []


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
        small_data = np.array([[1, 1, 1], [1, 2, 1], [2, 3, 1]])
        self.input_df = pd.DataFrame(tabular_data, columns=column_names)
        self.small_input_df = pd.DataFrame(small_data, columns=column_names)
        self.x_df = self.small_input_df.drop(["rating"], axis=1)

    def test_split_data(self):
        base = DummyPreprocessor(data_df=self.input_df)
        train_X, test_X, train_y,  test_y = base.split_data(base.get_x(), base.get_y(), 0.2)
        assert train_X.shape[0] == 8
        assert train_y.shape[0] == 8
        assert test_X.shape[0] == 2
        assert test_y.shape[0] == 2

    def test_transform_numerical(self):
        sol = np.array([[1, 1, 1], [1, 2, 1], [2, math.log(float(3)) ** 2, 1]])
        base = DummyPreprocessor(data_df=self.small_input_df)
        base.transform_numerical()
        assert base.data_df.shape == (3, 3)
        assert np.array_equal(sol, base.data_df.values)

    def test_transform_categorical(self):
        sol = np.array([[0, 1, 1], [0, 2, 1], [1, 3, 1]])
        base = DummyPreprocessor(data_df=self.small_input_df)
        base.transform_categorical()
        assert base.data_df.shape == (3, 3)
        assert np.array_equal(sol, base.data_df.values)

    def test_get_hash_size(self):
        base = DummyPreprocessor(data_df=self.small_input_df)
        base.transform_categorical()
        assert base.get_hash_size() == [2]

    def test_get_x(self):
        sol = self.x_df
        base = DummyPreprocessor(data_df=self.small_input_df)
        pd.testing.assert_frame_equal(sol, base.get_x())

    def test_get_x_numerical(self):
        sol = self.x_df[['num_people']].values
        base = DummyPreprocessor(data_df=self.small_input_df)
        assert np.array_equal(base.get_x_numerical(
            self.x_df), sol)

    def test_get_x_categorical(self):
        sol = self.x_df[['user_id']].values
        base = DummyPreprocessor(data_df=self.small_input_df)
        assert np.array_equal(base.get_x_categorical(
            self.x_df), sol)

    def test_get_y(self):
        sol = np.ones(3)
        base = DummyPreprocessor(data_df=self.small_input_df)
        assert np.array_equal(base.get_y(), sol)

    def test_get_categorical_count(self):
        base = DummyPreprocessor(data_df=self.small_input_df)
        assert base.get_categorical_count() == 1

    def test_get_numerical_count(self):
        base = DummyPreprocessor(data_df=self.small_input_df)
        assert base.get_numerical_count() == 1

    def test_MovielensPreprocessor(self):
        movielens = MovielensPreprocessor(csv_path=os.path.join(
            dataset_directory, 'movielens/ratings-10k.dat'))
        movielens.preprocess()
        assert movielens.data_df.shape == (10000, 3)

    def test_CriteoPreprocessor(self):
        criteo = CriteoPreprocessor(csv_path=os.path.join(
            dataset_directory, 'criteo/train-10k.txt'))
        criteo.preprocess()
        assert criteo.data_df.shape == (10000, 40)

    def test_NetflixPreprocessor(self):
        netflix = NetflixPrizePreprocessor(non_csv_path=os.path.join(
            dataset_directory, 'netflix/combined_data_1-10k.txt'), csv_path=os.path.join(dataset_directory, 'netflix/combined_data_1-10k.csv'))
        netflix.preprocess()
        assert netflix.data_df.shape == (10000, 3)

    def test_AvazuPreprocessor(self):
        avazu = AvazuPreprocessor(csv_path=os.path.join(
            dataset_directory, 'avazu/train-10k'))
        avazu.preprocess()
        assert avazu.data_df.shape == (9999, 23)
