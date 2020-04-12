# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from typing import List

from sklearn.model_selection import train_test_split
from collections import defaultdict

import pandas as pd
import numpy as np
import sys
import time
import gc
import psutil
import os


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
    for rating prediction recommendation methods
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


class BaseCTRPreprocessor(BaseProprocessor):
    """
    for click-through rate (CTR) recommendation methods
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


class CriteoPreprocessor(BaseCTRPreprocessor):

    def __init__(self, dataset_path, save_path):
        super(CriteoPreprocessor, self).__init__(dataset_path=dataset_path, )
        self.save_path = save_path
        self.label_num = 1
        self.numer_num = 13
        self.categ_num = 26
        self.categ_filter = 10

        self.label_names = ["l" + str(i) for i in range(self.label_num)]
        self.numer_names = ["d" + str(i) for i in range(self.numer_num)]
        self.categ_names = ["c" + str(i) for i in range(self.categ_num)]

        self.columns_names = self.label_names + self.numer_names + self.categ_names
        self.used_columns_names = self.label_names + self.numer_names + self.categ_names
        self.dtype_dict = {n: np.float32 for n in self.used_columns_names}

        self._load_data()

    def _load_data(self):

        """
        :return:

        Criteo dataset has 40 features per line, where [0] = label, [1-13] = numerical, and [14-39] = categorical
        """
        label_ll = [list() for _ in range(self.label_num)]  # list of label data
        numer_ll = [list() for _ in range(self.numer_num)]  # list of each numerical feature's data
        categ_ll = [list() for _ in range(self.categ_num)]  # list of each categorical feature's data
        # Define dictionary where key="feature index" & val="dictionary" where key="category" & val="count".
        categ_count_dict = defaultdict(lambda: defaultdict(int))


        # Step 1: Load label, numerical, and categorical features
        with open(self.dataset_path, 'r') as ld:
            label_start = 0
            numer_start = 0 + self.label_num
            categ_start = 0 + self.label_num + self.numer_num

            for line in ld:
                line = [v if v != "" else -1 for v in line.strip('\n').split('\t')]  # replace missing value with -1

                for i in range(self.label_num):  # record label feature data
                    label_ll[i].append(int(line[label_start + i]))
                for i in range(self.numer_num):  # record numerical feature data
                    numer_ll[i].append(int(line[numer_start + i]))
                for i in range(self.categ_num):  # record categorical feature data
                    categ_ll[i].append(line[categ_start + i])
                    categ_count_dict[i][line[categ_start + i]] += 1  # count category occurrences

        # Step 2: Generate dictionary for renumbering categories.
        # Define dictionary where key="feature index" & val="dictionary" where key="category" & val="index".
        categ_index_dict = defaultdict(lambda: defaultdict(int))

        for feat_index, categ_dict in categ_count_dict.items():
            categ_index = 1  # reserve 0 for filtered infrequent categories, which will be indexed later
            for categ, count in categ_dict.items():
                if count >= self.categ_filter:
                    categ_index_dict[feat_index][categ] = categ_index
                    categ_index += 1

        del categ_count_dict  # release memory
        gc.collect()

        # Step 3: Reindex categories and obtain hash statistics.
        for i in range(self.categ_num):
            categ_ll[i] = [categ_index_dict[i][c] for c in categ_ll[i]]  # filtered categories will be indexed 0

        self.hash_sizes = [len(categ_index_dict[i].values())+1 for i in range(self.categ_num)]  # +1 compensates index 0

        del categ_index_dict  # release memory
        gc.collect()

        # Step 4: Format data.
        array_data = np.concatenate((np.asarray(label_ll).T, np.asarray(numer_ll).T, np.asarray(categ_ll).T), axis=1)

        del label_ll  # release memory
        del numer_ll
        del categ_ll
        gc.collect()

        # TODO: Support save/load preprocessed data.
        np.save(self.save_path, array_data)

        self.pd_data = pd.DataFrame(array_data, columns=self.used_columns_names)
        del array_data
        gc.collect()

        for col_name, dtype in self.dtype_dict.items():
            self.pd_data[col_name] = self.pd_data[col_name].astype(dtype)

    def preprocessing(self, test_size, random_state):
        self.X = self.pd_data.iloc[::, 1:].values
        self.y = self.pd_data.iloc[::, 1].values

        train_X, val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, test_size=test_size,
                                                                              random_state=random_state)
        # Reformat numerical features and categorical features.
        self.train_X = [train_X[:, :self.numer_num], train_X[:, self.numer_num:]]

        del train_X  # release memory
        gc.collect()

        self.val_X = [val_X[:, :self.numer_num], val_X[:, self.numer_num:]]

        del val_X  # release memory
        gc.collect()



class NetflixPrizePreprocessor(BaseRatingPredictionProprocessor):

    def __init__(self, dataset_path):
        super(NetflixPrizePreprocessor, self).__init__(dataset_path=dataset_path, )
        self.columns_names = ["user_id", "item_id", "rating"]
        self.used_columns_names = ["user_id", "item_id", "rating"]
        self.dtype_dict = {"user_id": np.int32, "item_id": np.int32, "rating": np.float32}
        self._load_data()

    def _load_data(self):
        """

        CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
        :return:
        """
        cols = [list() for _ in range(len(self.columns_names))]
        user2index_dict = dict()  # needs renumber since CustomerIDs range from 1 to 2649429, with gaps

        for fp in self.dataset_path:  # TODO: deal with list of paths

            with open(fp, 'r') as f:
                for line in f.readlines():
                    if ':' in line:
                        movie = line.strip(":\n")  # needs no renumber since MovieIDs range from 1 to 17770 sequentially
                    else:
                        user, rating, _ = line.strip().split(',')
                        cols[1].append(movie)
                        cols[2].append(rating)

                        if user in user2index_dict.keys():
                            cols[0].append(user2index_dict[user])
                        else:
                            cols[0].append(len(user2index_dict.keys()))  # number users from 0
                            user2index_dict[user] = len(user2index_dict.keys())

        self.pd_data = pd.DataFrame(dict(zip(self.columns_names, cols)))

        for col_name, dtype in self.dtype_dict.items():
            self.pd_data[col_name] = self.pd_data[col_name].astype(dtype)

        self.pd_data = self.pd_data[self.used_columns_names]
        self.user_num = len(set(cols[0]))
        self.item_num = len(set(cols[1]))


    def preprocessing(self, test_size, random_state):
        self.X = self.pd_data.iloc[::, :-1].values
        self.y = self.pd_data.iloc[::, -1].values
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, test_size=test_size,
                                                                              random_state=random_state)


class MovielensPreprocessor(BaseRatingPredictionProprocessor):

    used_columns_names: List[str]

    def __init__(self, dataset_path, sep='::'):
        super(MovielensPreprocessor, self).__init__(dataset_path=dataset_path, )
        self.columns_names = ["user_id", "item_id", "rating", "timestamp"]
        self.used_columns_names = ["user_id", "item_id", "rating"]
        self.dtype_dict = {"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "timestamp": np.int32}
        self.sep = sep
        self._load_data()

    def _load_data(self):
        self.pd_data = pd.read_csv(self.dataset_path, sep=self.sep, header=None, names=self.columns_names,
                                   dtype=self.dtype_dict)
        self.pd_data = self.pd_data[self.used_columns_names]

    def preprocessing(self, test_size, random_state):
        self.X = self.pd_data.iloc[::, :-1].values
        self.y = self.pd_data.iloc[::, -1].values
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, test_size=test_size,
                                                                              random_state=random_state)


class MovielensCTRPreprocessor(BasePointWiseProprocessor):

    def __init__(self, dataset_path, sep='::'):
        super(MovielensCTRPreprocessor, self).__init__(dataset_path=dataset_path)
        self.columns_names = ["user_id", "item_id", "rating", "timestamp"]
        self.used_columns_names = ["user_id", "item_id", "rating"]
        self.dtype_dict = {"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "timestamp": np.int32}
        self.sep = sep
        self._load_data()

    def _load_data(self):
        self.pd_data = pd.read_csv(self.dataset_path, sep=self.sep, header=None, names=self.columns_names,
                                   dtype=self.dtype_dict)
        self.pd_data = self.pd_data[self.used_columns_names]

    def _negative_sampling(self, input_df, num_neg, mult=1):
        """ Add a column of negative item IDs to the input DataFrame
        :param input_df: DataFrame user-item interactions with columns=[user_id, item_id, rating]
        :param num_neg: Integer number of negative interaction to sample per user-item interaction (cannot be 0)
        :param mult: Integer multiplier for boosting the rank of negative item candidates in case of deficiency
        :return: DataFrame user-item interactions with columns=[user_id, item_id, rating, neg_item_ids]
        TODO: proper mult should be calculated by program and not asking user to specify
        TODO: notice a positive relation may receive duplicated negatively sampled items; consider this universal but may require attention
        """
        # Find candidate negative items (CNI) for each user
        item_set = set(input_df['item_id'].unique())
        u_pi_series = input_df.groupby('user_id')['item_id'].apply(set)  # series where index=user & data=positive items
        u_cni_series = item_set - u_pi_series  # series where index=user & data=candidate negative items

        # Find sampled negative items (SNI) for each user
        u_sni_series = u_cni_series.agg(  # series where index=user & data=sampled negative items
            lambda x: np.random.RandomState().permutation(list(x)*num_neg*mult)[:num_neg * (len(item_set) - len(x))])

        # Distribute SNI to positive user-item interactions by chunk
        u_snic_series = u_sni_series.agg(  # series where index=user & data=sampled negative item chunks (SNIC)
            lambda x: [x[i*num_neg: (i+1)*num_neg] for i in range(int(len(x)/num_neg))])

        # Distribute SNIC to users
        u_snic_df = u_snic_series.to_frame().apply(pd.Series.explode).reset_index()

        # Add SNI to input DataFrame
        input_df["neg_item_ids"] = u_snic_df["item_id"]

        return input_df

    def preprocessing(self, test_size, num_neg, random_state, mult):
        compact_data = self._negative_sampling(self.pd_data, num_neg=num_neg, mult=mult)
        compact_X = compact_data.loc[:, compact_data.columns != 'rating']
        compact_y = compact_data[['rating']]
        compact_train_X, compact_val_X, compact_train_y, compact_val_y = train_test_split(compact_X, compact_y,
            test_size=test_size, random_state=random_state)

        def expand(unexpanded_X, unexpanded_y):
            # Extract positive relations
            unexpanded_data = pd.concat([unexpanded_X, unexpanded_y], axis=1)
            pos_data = unexpanded_data[['user_id', 'item_id', 'rating']]
            pos_data['rating'] = 1

            # Expand negative relations
            neg_data = unexpanded_data.explode('neg_item_ids').dropna()[['user_id', 'neg_item_ids', 'rating']].rename(
                columns={'neg_item_ids': 'item_id'})
            neg_data['rating'] = 0

            # Combine negative relations with positive relations
            pos_neg_data = pos_data.append(neg_data, ignore_index=True).reset_index(drop=True)

            pos_neg_data["item_id"] = pos_neg_data["item_id"].astype(np.int32)
            pos_neg_data["user_id"] = pos_neg_data["user_id"].astype(np.int32)
            pos_neg_data["rating"] = pos_neg_data["rating"].astype(np.int32)

            return pos_neg_data

        self.pd_data = expand(compact_X, compact_y)
        self.X = (self.pd_data.loc[:, self.pd_data.columns != 'rating']).values
        self.y = self.pd_data['rating'].values

        expanded_train_data = expand(compact_train_X, compact_train_y)
        self.train_X = (expanded_train_data.loc[:, expanded_train_data.columns != 'rating']).values
        self.train_y = expanded_train_data['rating'].values

        expanded_val_data = expand(compact_val_X, compact_val_y)
        self.val_X = (expanded_val_data.loc[:, expanded_val_data.columns != 'rating']).values
        self.val_y = expanded_val_data['rating'].values
