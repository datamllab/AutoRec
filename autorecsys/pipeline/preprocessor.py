# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from autorecsys.utils import load_config


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


class Movielens1MPreprocessor(BaseProprocessor):

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


class Movielens1MCTRPreprocessor(BaseProprocessor):

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


def negative_sampling(input_df, num_neg, seed=1314):
    """ Negative sampling without replacement
    :param input_df: DataFrame user-item interaction
    :param num_neg: Integer number of negative interaction to sample per user-item interaction
    :param seed: Integer seed for random sampling
    :return: DataFrame user-item positive and negative sampling
    """
    # Perform negative sampling
    input_df["item_id"] = input_df["item_id"].astype(np.int32)
    input_df["user_id"] = input_df["user_id"].astype(np.int32)
    input_df["user_id"] = input_df["user_id"].astype(np.float32)
    item_set = set(input_df['item_id'].unique())
    user_pos_items_series = input_df.groupby('user_id')['item_id'].apply(set)
    user_neg_items_series = item_set - user_pos_items_series
    user_sampled_neg_items_series = user_neg_items_series.agg(
        lambda x: np.random.RandomState(seed=seed).permutation(list(x))[:num_neg * (len(item_set) - len(x))])

    # Convert negative samples to have input format
    user_sampled_neg_items_df = user_sampled_neg_items_series.to_frame().explode('item_id').dropna()
    user_sampled_neg_items_df['rating'] = 0.
    user_sampled_neg_items_df.reset_index(inplace=True)

    # Combine positive and negative samples
    input_df["rating"] = 1.
    output_df = input_df.append(user_sampled_neg_items_df, ignore_index=True).reset_index(drop=True)
    output_df = output_df.sort_values(by=['user_id']).reset_index(drop=True)

    return output_df


def data_load_from_config(config=None):
    if config is None:
        config = "./examples/configs/data_default_config.yaml"
    data_config = load_config(config)["DataOption"]
    data_config["dtype"] = {key: eval(data_config["dtype"][key]) for key in data_config["dtype"]}
    train_X, train_y, val_X, val_y = data_load(**data_config)
    return train_X, train_y, val_X, val_y


def data_load(dataset="movielens", dataset_path=None, col_names=None, dtype=None, used_col_names=None, test_size=0.1):
    if dataset == "movielens":
        X, y = data_load_movielens(dataset_path, col_names, used_col_names, dtype)
    else:
        raise ValueError("Embedding_dim should be a string")
    train_X, train_y, val_X, val_y = data_split(X, y, test_size)
    return train_X, train_y, val_X, val_y


def data_load_movielens(dataset_path, col_names, used_col_names=None, dtype=None):
    if used_col_names == None:
        used_col_names = col_names
    pd_data = pd.read_csv(dataset_path, sep="::", header=None, names=col_names, dtype=dtype)
    pd_data = pd_data[used_col_names]
    X = pd_data.iloc[::, :-1].values
    y = pd_data.iloc[::, -1].values
    return X, y


def data_split(X, y, test_size=0.1):
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=test_size, random_state=0)
    return train_X, train_y, val_X, val_y


def test_data_load_from_config():
    # data_load( dataset= "movielens", dataset_path ="./examples/datasets/ml-1m/ratings.dat", col_names = ["user_id", "item_id", "rating", "timestamp"], used_col_names = ["user_id", "item_id", "rating"] ,dtype={"user_id":np.int32, "item_id":np.int32, "rating":np.float32, "timestamp":np.int32}  )
    config_file = "./examples/configs/data_default_config.yaml"
    # data_config = load_config(config_file)[ "DataOption"  ]
    # data_config[ "dtype" ] = { key : eval( data_config[ "dtype" ][key] )  for key in data_config[ "dtype" ]  }
    # print( data_config )
    data_load_from_config(config_file)


def test_Movielens1MPreprocessor():
    ml_1m = Movielens1MPreprocessor("./tests/datasets/ml-1m/ratings.dat")
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


def test_Movielens1MCTRPreprocessor():
    ml_1m = Movielens1MCTRPreprocessor("./tests/datasets/ml-1m/ratings.dat")
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


if __name__ == "__main__":
    test_Movielens1MPreprocessor()
    test_Movielens1MCTRPreprocessor()
