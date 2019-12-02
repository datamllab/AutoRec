# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

class BaseProprocessor(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, dataset_path=None, train_path=None, val_path=None, test_size=None):
        super(BaseProprocessor, self).__init__()
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.val_path = val_path

    @abstractmethod
    def preprocessing(self, **kwargs):
        raise NotImplementedError


class BaseRatingPredictionProprocessor(BaseProprocessor):
    """
    for RatingPrediction recommendation methods, rating prediction for Movielens
    and can also for the similar dataset for rating prediction task
    """

    @abstractmethod
    def __init__(self, dataset_path=None, train_path=None, val_path=None, test_size=None):
        super(BaseProprocessor, self).__init__()
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.val_path = val_path

    @abstractmethod
    def preprocessing(self, **kwargs):
        raise NotImplementedError


class BasePointWiseProprocessor(BaseProprocessor):
    """
    for PointWise recommendation methods, CTR
    for PointWise recommendation methods, rating prediction for Movielens
    and can also for the similar dataset

    """

    @abstractmethod
    def __init__(self, dataset_path=None, train_path=None, val_path=None, test_size=None):
        super(BaseProprocessor, self).__init__()
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.val_path = val_path

    @abstractmethod
    def preprocessing(self, **kwargs):
        raise NotImplementedError


class BasePairWiseProprocessor(BaseProprocessor):
    """
    for PairWise recommendation methods
    """

    @abstractmethod
    def __init__(self, dataset_path=None, train_path=None, val_path=None, test_size=None):
        super(BaseProprocessor, self).__init__()
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.val_path = val_path

    @abstractmethod
    def preprocessing(self, **kwargs):
        raise NotImplementedError


class Movielens1MPreprocessor(BaseRatingPredictionProprocessor):

    def __init__(self, dataset_path):
        super(Movielens1MPreprocessor, self).__init__(dataset_path=dataset_path, )
        self.columns_names = ["user_id", "item_id", "rating", "timestamp"]
        self.used_columns_names = ["user_id", "item_id", "rating"]
        self.dtype_dict = {"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "timestamp": np.int32}
        self._load_data()

    def _load_data(self):
        self.pd_data = pd.read_csv(self.dataset_path, sep="::", header=None, names=self.columns_names,
                                   dtype=self.dtype_dict)
        self.pd_data = self.pd_data[self.used_columns_names]

    def preprocessing(self, test_size, random_state):
        self.X = self.pd_data.iloc[::, :-1].values
        self.y = self.pd_data.iloc[::, -1].values
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, test_size=test_size,
                                                                              random_state=random_state)


class Movielens1MCTRPreprocessor(BasePointWiseProprocessor):

    def __init__(self, dataset_path):
        super(Movielens1MCTRPreprocessor, self).__init__(dataset_path=dataset_path)
        self.columns_names = ["user_id", "item_id", "rating", "timestamp"]
        self.used_columns_names = ["user_id", "item_id", "rating"]
        self.dtype_dict = {"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "timestamp": np.int32}
        self._load_data()

    def _load_data(self):
        self.pd_data = pd.read_csv(self.dataset_path, sep="::", header=None, names=self.columns_names,
                                   dtype=self.dtype_dict)
        self.pd_data = self.pd_data[self.used_columns_names]

    def _negative_sampling(self, input_df, num_neg, seed=42):
        """ Negative sampling with replacement
        :param input_df: DataFrame user-item interactions with columns=[user_id, item_id, rating]
        :param num_neg: Integer number of negative interaction to sample per user-item interaction
        :param seed: Integer seed for random sampling
        :return: DataFrame user-item interactions with columns=[user_id, item_id, rating, neg_item_ids]
        """
        # Find negative sampling candidate items
        items = set(input_df['item_id'].unique())
        user_pos_items = input_df.groupby('user_id')['item_id'].apply(set)
        user_neg_items = items - user_pos_items
        input_df['neg_item_ids'] = input_df['user_id'].map(user_neg_items)

        # Perform negative sampling
        s_time = time.time()
        input_df['neg_item_ids'] = input_df['neg_item_ids'].agg(
            lambda x: np.random.RandomState(seed=seed).permutation(list(x))[:num_neg])
        e_time = time.time()
        print(f'use {e_time - s_time} seconds')
        input('\n\ntest ability to pass above code\n\n')
        return input_df

    def preprocessing(self, test_size, num_neg, random_state):
        compact_data = self._negative_sampling(self.pd_data[:self.pd_data.shape[0]//10], num_neg=num_neg, seed=random_state)
        compact_X = compact_data.iloc[:, self.pd_data.columns != 'rating']
        compact_y = compact_data['rating'].to_frame()
        compact_train_X, compact_val_X, compact_train_y, compact_val_y = train_test_split(compact_X, compact_y,
            test_size=test_size, random_state=random_state)

        def expand(c_X, c_y):
            c_data = pd.concat([c_X, c_y], axis=1)

            # Expand negative relations
            neg_data = c_data.explode('neg_items').dropna()[['user_id', 'neg_items', 'rating']].rename(
                columns={'neg_items': 'item_id'})
            neg_data['rating'] = 0

            # Combine with positive relations
            pos_data = c_data[['user_id', 'item_id', 'rating']]
            pos_data['rating'] = 1
            pos_neg_data = pos_data.append(neg_data, ignore_index=True).reset_index(drop=True)

            return pos_neg_data

        self.pd_data = expand(compact_X, compact_y)
        self.X = self.pd_data.iloc[:, self.pd_data.columns != 'rating']
        self.y = self.pd_data['rating'].to_frame()

        expanded_train_data = expand(compact_train_X, compact_train_y)
        self.train_X = expanded_train_data.iloc[:, self.pd_data.columns != 'rating']
        self.train_y = expanded_train_data['rating'].to_frame()

        expanded_val_data = expand(compact_val_X, compact_train_y)
        self.val_X = expanded_val_data.iloc[:, self.pd_data.columns != 'rating']
        self.val_y = expanded_val_data['rating'].to_frame()


        # Need to set self.pd_data, self.X, self.y, self.train_X, self.val_X, self.train_y, self.val_y


class TabularPreprocessor(BaseProprocessor):
    def __init__(self, config):
        super(TabularPreprocessor, self).__init__(config)
