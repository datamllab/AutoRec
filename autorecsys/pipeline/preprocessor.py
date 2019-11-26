# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from autorecsys.utils import load_config


class BaseProprocessor(metaclass=ABCMeta):
    def __init__(self, config, **kwarg):
        super(BaseProprocessor, self).__init__()
        self.config = config

    @abstractmethod
    def load_data(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def get_batch(self):
        raise NotImplementedError

class TabularPreprocessor(BaseProprocessor):
    def __init__(self, config):
        super(TabularPreprocessor, self).__init__(config)


def negative_sampling(input_df, num_neg, seed=42):
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
        lambda x: np.random.RandomState(seed=seed).permutation(list(x))[:num_neg * (len(item_set)-len(x))])

    # Convert negative samples to have input format
    user_sampled_neg_items_df = user_sampled_neg_items_series.to_frame().explode('item_id').dropna()
    user_sampled_neg_items_df['rating'] = 0
    user_sampled_neg_items_df.reset_index(inplace=True)

    # Combine positive and negative samples
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


def test():
    # data_load( dataset= "movielens", dataset_path ="./examples/datasets/ml-1m/ratings.dat", col_names = ["user_id", "item_id", "rating", "timestamp"], used_col_names = ["user_id", "item_id", "rating"] ,dtype={"user_id":np.int32, "item_id":np.int32, "rating":np.float32, "timestamp":np.int32}  )
    config_file = "./examples/configs/data_default_config.yaml"
    # data_config = load_config(config_file)[ "DataOption"  ]
    # data_config[ "dtype" ] = { key : eval( data_config[ "dtype" ][key] )  for key in data_config[ "dtype" ]  }
    # print( data_config )
    data_load_from_config(config_file)


if __name__ == "__main__":
    test()
