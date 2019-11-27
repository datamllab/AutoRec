# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


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
        """ Negative sampling without replacement
        :param input_df: DataFrame user-item interaction
        :param num_neg: Integer number of negative interaction to sample per user-item interaction
        :param seed: Integer seed for random sampling
        :return: DataFrame user-item positive and negative sampling
        """
        # Perform negative sampling

        item_set = set(input_df['item_id'].unique())
        user_pos_items_series = input_df.groupby('user_id')['item_id'].apply(set)
        user_neg_items_series = item_set - user_pos_items_series
        user_sampled_neg_items_series = user_neg_items_series.agg(
            lambda x: np.random.RandomState(seed=seed).permutation(list(x))[:num_neg * (len(item_set) - len(x))])

        # Convert negative samples to have input format
        user_sampled_neg_items_df = user_sampled_neg_items_series.to_frame().explode('item_id').dropna()
        user_sampled_neg_items_df['rating'] = 0
        user_sampled_neg_items_df.reset_index(inplace=True)

        # Combine positive and negative samples
        input_df["rating"] = 1
        output_df = input_df.append(user_sampled_neg_items_df, ignore_index=True).reset_index(drop=True)
        output_df = output_df.sort_values(by=['user_id']).reset_index(drop=True)

        #set data type
        output_df["item_id"] = output_df["item_id"].astype(np.int32)
        output_df["user_id"] = output_df["user_id"].astype(np.int32)
        output_df["rating"] = output_df["rating"].astype(np.int32)
        return output_df

    def preprocessing(self, test_size, num_neg, random_state):
        self.pd_data = self._negative_sampling(self.pd_data, num_neg=num_neg, seed=random_state)
        self.X = self.pd_data.iloc[::, :-1].values
        self.y = self.pd_data.iloc[::, -1].values
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, test_size=test_size,
                                                                              random_state=random_state)

class TabularPreprocessor(BaseProprocessor):
    def __init__(self, config):
        super(TabularPreprocessor, self).__init__(config)