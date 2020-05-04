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
import math
from pathlib import Path
from autorecsys.utils.common import load_pickle, save_pickle


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


class AvazuPreprocessor(BaseCTRPreprocessor):

    def __init__(self, dataset_path="./examples/datasets/avazu/sampled_train_10000.txt"):
        super(AvazuPreprocessor, self).__init__(dataset_path=dataset_path, )
        # /home/thwang1231/autorec/examples/datasets/avazu/sampled_train_10000.txt  # 5/4 loss: 0.6239
        # /home/thwang1231/autorec/examples/datasets/avazu/sampled_train_2mil.txt   # 5/4 loss: 0.3718
        # /home/thwang1231/autorec/examples/datasets/avazu/train                    # 5/4 loss: 0.3859
        # Step 1: Set attributes used/set during dataset loading & data preprocessing.
        self.has_header = True
        self.column_names = None
        self.fit_dictionary_path = "./avazu_fit_dictionary.pkl"
        self.transformed_data_path = "./avazu_transformed_data.pkl"
        self.categ_filter = 3

        # Set indices variables
        # TODO: fix the situation that these indices are relative to the raw data and not to the preprocessed data
        self.ignored_indices = [0]
        self.label_indices = [1]
        self.numer_indices = []
        # TODO: consider this is actually "used" categorical indices
        self.categ_indices = [i for i in range(24) if i not in self.label_indices + self.ignored_indices]

        self.label_num = len(self.label_indices)
        self.numer_num = len(self.numer_indices)
        self.categ_num = len(self.categ_indices)

        # Set variables that will be set in load_data()
        # TODO: Remark although Avazu datast contains column header, and we do support header loading, we assume some
        #   info about benchmark datasets, e.g., like Avazu, Criteo, Movielens, and Netflix, are simply known. Determine
        #   which of these info's corresponding instance variables are and pre-set them with constant values.
        self.column_names = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
                             'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model',
                            'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
        self.used_column_names = None
        self.pd_data = None
        self.hash_sizes = None
        self.dtype_dict = None

        # Remark we no longer use label_num = len(label_indices)
        # Remark we no longer use categ_num = len(categ_indices)

        # self.used_columns_names = self.columns_names
        # self.dtype_dict = {n: np.float32 for n in self.used_columns_names}


    def load_data(self):
        """
        Set the following variables:
            1) self.used_column_names: set the variable here because column_names may be loaded from dataset header
            2) self.pd_data: set the variable here because the dataframe contain loaded and transformed data
            3) self.hash_sizes: set the variable here because it is inferred from pd_data
                    Note: do not infer from fit_dictionary because there might not such file to load.
                    On the other hand, transformed data is always available at the end of the file so infer from that
            4) self.dtype_dict: set the variable because we use used_column_name as key, which can be loaded here
        :return:
        """
        # Step 0: Load transformed data if it exists.
        if Path(self.transformed_data_path).is_file():  # found transformed data
            self.pd_data = load_pickle(self.transformed_data_path)
            print("col names:", self.column_names)
            print("cat indices:", self.categ_indices)
            categ_names = [self.column_names[i] for i in self.categ_indices]
            self.hash_sizes = [self.pd_data[categ_name].nunique() for categ_name in categ_names]
            return

        # Step 1: Load data.
        label_ld = defaultdict(list)  # where {k: v}={col index: list of labels}
        categ_ld = defaultdict(list)  # where {k: v}={col index: list of categories}
        # TODO: changed data structure holding loaded data, make the same change to other classes
        categ_count_dict = defaultdict(lambda: defaultdict(int))  # where {k1: {k2, v}}={col index: {category: count}}

        with open(self.dataset_path, 'r') as ld:
            if self.has_header:
                # TODO: parameterizable delimiter (or at least make delimiter a passable argument at parent class)
                self.column_names = ld.readline().strip('\n').split(',')
            for line in ld:
                # TODO: make placeholder value a passalbe argument at parent class
                line = [v if v != "" else "0" for v in line.strip('\n').split(',')]

                for i in self.label_indices:
                    label_ld[i].append(int(line[i]))  # typecast from string to save memory
                for i in self.categ_indices:
                    categ_ld[i].append(line[i])  # cannot typecast string data
                    categ_count_dict[i][line[i]] += 1

        # Step 2: Creat fit dictionary.
        if Path(self.fit_dictionary_path).is_file():
            categ_index_dict = load_pickle(self.fit_dictionary_path)
        else:
            categ_index_dict = defaultdict(lambda: defaultdict(int))

            for i, categ_dict in categ_count_dict.items():
                categ_index = 0
                for categ, count in categ_dict.items():
                    if count > self.categ_filter:
                        categ_index_dict[i][categ] = categ_index
                        categ_index += 1
            del categ_count_dict
            gc.collect()
            # Save fit dictionary.
            save_pickle(self.fit_dictionary_path, dict(categ_index_dict))  # cannot pickle defaultdict using lambda

        # Step 3: Transform categorical data.
        for i, categ_l in categ_ld.items():
            for j, c in enumerate(categ_l):
                if c in categ_index_dict[i]:
                    categ_ld[i][j] = categ_index_dict[i][c]
                else:
                    categ_ld[i][j] = len(categ_index_dict[i])

        # Step 4: Format data and obtain hash statistics
        categ_ll = np.asarray([categ_ld[i] for i in self.categ_indices])
        label_ll = np.asarray([label_ld[i] for i in self.label_indices])

        # TODO: concatenate columns by absolute reference instead of relative
        array_data = np.concatenate((label_ll.T, categ_ll.T), axis=1)
        del categ_ll, label_ll
        gc.collect()

        # TODO: Note these variables (other than fit_dict & pd_data) are not saved. Please consider which variables can
        #   be inferred downstream and thus do no have to be saved, and which cannot and thus have to be saved.
        self.used_column_names = [self.column_names[i] for i in self.label_indices + self.categ_indices]
        self.pd_data = pd.DataFrame(array_data, columns=self.used_column_names)
        del array_data
        gc.collect()

        categ_names = [self.column_names[i] for i in self.categ_indices]
        self.hash_sizes = [self.pd_data[categ_name].nunique() for categ_name in categ_names]
        self.dtype_dict = {n: np.float32 for n in self.used_column_names}
        for col_name, dtype in self.dtype_dict.items():
            self.pd_data[col_name] = self.pd_data[col_name].astype(dtype)

        # Save transformed data.
        save_pickle(self.transformed_data_path, self.pd_data)

    def split_data(self, X, y, train_size, validate_size):
        """

        :param X: numpy array
        :param y: numpy array
        :param train_size:
        :param validate_size:
        :return:
        """
        # Step 1: Obtain shuffled data indices.
        data_size = X.shape[0]
        shuffled_indices = np.random.permutation(range(data_size))

        # Step 2: Determine data split positions.
        train_end = int(train_size * data_size)
        validate_end = train_end + int(validate_size * data_size)

        # Step 3: Split shuffled data.
        train_X = X[shuffled_indices][:train_end]
        train_y = y[shuffled_indices][:train_end]
        validate_X = X[shuffled_indices][train_end:validate_end]
        validate_y = y[shuffled_indices][train_end:validate_end]
        test_X = X[shuffled_indices][validate_end:]
        test_y = y[shuffled_indices][validate_end:]

        return train_X, train_y, validate_X, validate_y, test_X, test_y

    def preprocessing(self, train_size, validate_size, random_state=42):
        # Step 1: Load data for fit and transform categorical data.
        self.load_data()

        # Step 2: Transform numerical data.
        # self.scale_numerical_data()

        # Step 3: Split data for training, validation, and testing.
        # TODO: reference by class variable and not constant
        X = self.pd_data.iloc[:, 1:].values  # because 0th column in raw data is ignored/not loaded
        y = self.pd_data.iloc[:, [0]].values
        train_X, train_y, validate_X, validate_y, test_X, test_y = self.split_data(
            X, y, train_size=train_size, validate_size=validate_size)

        # Step 4: Arrange data for training algorithms.
        # print(train_X[:, 1:].shape)
        # print(type(train_X))
        # filler = np.asarray([list() for _ in range(train_X.shape[0])])
        # print(filler.shape)
        # input()

        # No need to slice as there is only categorical feature, i.e., no numerical feature
        # train_X = [train_X[:, 1:]]
        # validate_X = [validate_X[:, 1:]]
        # test_X = [test_X[:, 1:]]

        return train_X, train_y, validate_X, validate_y, test_X, test_y


class CriteoPreprocessor(BaseCTRPreprocessor):

    def __init__(self, dataset_path="./examples/datasets/criteo_sample_10000/train_examples.txt"):
        # TODO: change dataset_path to package-firendly path
        # dataset_path = "./examples/datasets/criteo_full/train.txt"
        # dataset_path = "./examples/datasets/criteo_sample_10000/train_examples.txt"
        # dataset_path = "./examples/datasets/criteo_2mil/train_2mil.txt"
        super(CriteoPreprocessor, self).__init__(dataset_path=dataset_path, )

        # Step 1: Set attributes used during dataset loading & data preprocessing.
        self.fit_dictionary_path = "./criteo_fit_dictionary.pkl"
        self.transformed_data_path = "./criteo_transformed_data.pkl"

        self.label_num = 1
        self.numer_num = 13
        self.categ_num = 26
        self.categ_filter = 10

        # TODO: E.g., set label_indices first and then infer label number is more logical.
        self.label_indices = np.arange(self.label_num)
        self.numer_indices = np.arange(self.numer_num) + self.label_num
        self.categ_indices = np.arange(self.categ_num) + self.label_num + self.numer_num

        self.label_names = ["l" + str(i) for i in range(self.label_num)]
        self.numer_names = ["n" + str(i) for i in range(self.numer_num)]
        self.categ_names = ["c" + str(i) for i in range(self.categ_num)]

        self.columns_names = self.label_names + self.numer_names + self.categ_names
        self.used_columns_names = self.label_names + self.numer_names + self.categ_names
        # Set data type of all columns as integer:
        #   1) The label feature is binary and thus expressible in floats.
        #   2) The numerical features are expressible in floats.
        #   3) The categorical features are indexed as numerical values and thus expressible in floats.
        self.dtype_dict = {n: np.float32 for n in self.used_columns_names}

        # Step 2: Set attributes used during data split.
        self.X = None
        self.y = None
        self.train_X = None
        self.train_y = None
        self.validate_X = None
        self.validate_y = None
        self.test_X = None
        self.test_y = None

    def load_data(self):

        """ Load raw Criteo data, in progress setting self.hash_sizes (from categorical data) and self.pd_data (from
            all data) and saving the fit dictionary and self.pd_data
        :return:

        Criteo dataset has 40 features per line, where [0] = label, [1-13] = numerical, and [14-39] = categorical
        TODO: allow to do fit and transform again even if files exist
        """
        # Step 0: Load transformed data if it exists.
        if Path(self.transformed_data_path).is_file():  # found transformed data
            self.pd_data = load_pickle(self.transformed_data_path)
            # TODO: Here we should also load used label, numerical, and categorical column names, meaning they should be
            #  saved. This allows:
            #       1) using data without prior knowledge of the mentioned class variables.
            #       2) rationalizing once we know something, we have the option to recover them without remembering them
            #       3) so no need to rely on any "self" variables, i.e., our saved data is self-sufficient
            #  dump() the class instance as a YAML config document
            self.hash_sizes = [self.pd_data[categ_name].nunique() for categ_name in self.categ_names]
            return

        # Step 1: Load label, numerical, and categorical features
        label_ll = [list() for _ in range(self.label_num)]  # list of label data
        numer_ll = [list() for _ in range(self.numer_num)]  # list of each numerical feature's data
        categ_ll = [list() for _ in range(self.categ_num)]  # list of each categorical feature's data
        # Define defaultdict where key="feature index" & val="defaultdict" where key="category" & val="count".
        categ_count_dict = defaultdict(lambda: defaultdict(int))

        with open(self.dataset_path, 'r') as ld:
            for line in ld:
                # Using zero "0" as substitute for missing values in:
                #   1) Numerical feature may corrupt data because 0 is numerical.
                #   2) Categorical feature will not corrupt data because the string "0" is a new category.
                # TODO: Need a principled way to determine missing values for numerical features.
                line = [v if v != "" else "0" for v in line.strip('\n').split('\t')]

                for i, j in enumerate(self.label_indices):  # record label feature data
                    label_ll[i].append(int(line[j]))  # typecast from string to save memory
                for i, j in enumerate(self.numer_indices):  # record numerical feature data
                    numer_ll[i].append(int(line[j]))  # typecast from string to save memory
                for i, j in enumerate(self.categ_indices):  # record categorical feature data
                    categ_ll[i].append(line[j])  # cannot typecast string data
                    categ_count_dict[i][line[j]] += 1  # count category occurrences

        # Step 2: Create dictionary for indexing categories.
        if Path(self.fit_dictionary_path).is_file():
            categ_index_dict = load_pickle(self.fit_dictionary_path)
        else:
            # Define defaultdict where key="feature index" & val="defaultdict" where key="category" & val="index".
            # TODO: Implement as a parent-class function
            categ_index_dict = defaultdict(lambda: defaultdict(int))

            # TODO: Rename variables, e.g., feat_index > i
            for feat_index, categ_dict in categ_count_dict.items():
                categ_index = 0
                for categ, count in categ_dict.items():
                    if count >= self.categ_filter:  # index filtered categories at a later stage
                        categ_index_dict[feat_index][categ] = categ_index
                        categ_index += 1
            del categ_count_dict  # release memory
            gc.collect()
            # Save fit dictionary.
            save_pickle(self.fit_dictionary_path, dict(categ_index_dict))  # cannot pickle defaultdict using lambda

        # Step 3: Index categories.
        for i, categ_l in enumerate(categ_ll):
            for j, c in enumerate(categ_l):
                if c in categ_index_dict[i]:
                    categ_ll[i][j] = categ_index_dict[i][c]  # index in-place
                else:  # index filtered categories as the end index
                    categ_ll[i][j] = len(categ_index_dict[i])
        del categ_index_dict  # release memory
        gc.collect()

        # Step 4: Format data and obtain hash statistics.
        array_data = np.concatenate((np.asarray(label_ll).T, np.asarray(numer_ll).T, np.asarray(categ_ll).T), axis=1)
        del label_ll, numer_ll, categ_ll  # release memory
        gc.collect()

        self.pd_data = pd.DataFrame(array_data, columns=self.used_columns_names)
        self.hash_sizes = [self.pd_data[categ_name].nunique() for categ_name in self.categ_names]
        del array_data  # release memory
        gc.collect()

        for col_name, dtype in self.dtype_dict.items():
            self.pd_data[col_name] = self.pd_data[col_name].astype(dtype)
        # Save transformed data.
        save_pickle(self.transformed_data_path, self.pd_data)

    def scale_numerical_data(self):

        # TODO: Designated transformation functions to specified place.
        def scale_by_natural_log(num):
            # TODO: 1) Explain why the conditional statement makes exception for numbers like 1, where 1>ln(1)**2
            if num > 2:
                num = math.log(float(num))**2
            return num

        for numer_name in self.numer_names:
            self.pd_data[numer_name] = self.pd_data[numer_name].map(scale_by_natural_log)

    def split_data(self, X, y, train_size, validate_size):
        """

        :param X: numpy array
        :param y: numpy array
        :param train_size:
        :param validate_size:
        :return:
        """
        # Step 1: Obtain shuffled data indices.
        data_size = X.shape[0]
        shuffled_indices = np.random.permutation(range(data_size))

        # Step 2: Determine data split positions.
        train_end = int(train_size * data_size)
        validate_end = train_end + int(validate_size * data_size)

        # Step 3: Split shuffled data.
        train_X = X[shuffled_indices][:train_end]
        train_y = y[shuffled_indices][:train_end]
        validate_X = X[shuffled_indices][train_end:validate_end]
        validate_y = y[shuffled_indices][train_end:validate_end]
        test_X = X[shuffled_indices][validate_end:]
        test_y = y[shuffled_indices][validate_end:]

        return train_X, train_y, validate_X, validate_y, test_X, test_y

    def preprocessing(self, train_size, validate_size, random_state=42):
        # Step 1: Load data for fit and transform categorical data.
        self.load_data()

        # Step 2: Transform numerical data.
        self.scale_numerical_data()

        # Step 3: Split data for training, validation, and testing.
        X = self.pd_data.iloc[:, 1:].values
        y = self.pd_data.iloc[:, [0]].values
        train_X, train_y, validate_X, validate_y, test_X, test_y = self.split_data(
            X, y, train_size=train_size, validate_size=validate_size)

        # Step 4: Arrange data for training algorithms.
        train_X = [train_X[:, :self.numer_num], train_X[:, self.numer_num:]]
        validate_X = [validate_X[:, :self.numer_num], validate_X[:, self.numer_num:]]
        test_X = [test_X[:, :self.numer_num], test_X[:, self.numer_num:]]

        return train_X, train_y, validate_X, validate_y, test_X, test_y

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
        user2index_dict = dict()  # needs renumber since CustomerID ranges from 1 to 2649429, with gaps

        for fp in self.dataset_path:  # TODO: deal with list of paths

            with open(fp, 'r') as f:
                for line in f.readlines():
                    if ':' in line:
                        movie = int(line.strip(":\n"))-1  # -1 because the sequential MovieIDs starts from 1
                    else:
                        user, rating = [int(v) for v in line.strip().split(',')[:2]]
                        cols[1].append(movie)
                        cols[2].append(rating)

                        if user in user2index_dict.keys():
                            cols[0].append(user2index_dict[user])
                        else:
                            cols[0].append(len(user2index_dict.keys()))  # number users from 0
                            user2index_dict[user] = len(user2index_dict.keys())

        # TODO: load date as well, later keep only selected data via self.used_columns_names
        self.pd_data = pd.DataFrame(dict(zip(self.columns_names, cols)))

        for col_name, dtype in self.dtype_dict.items():
            self.pd_data[col_name] = self.pd_data[col_name].astype(dtype)

        self.pd_data = self.pd_data[self.used_columns_names]
        self.user_num = self.pd_data["user_id"].nunique()
        self.item_num = self.pd_data["item_id"].nunique()

    def preprocessing(self, val_test_size, random_state):
        self.X = self.pd_data.iloc[::, :-1].values
        self.y = self.pd_data.iloc[::, [-1]].values
        self.train_X, self.val_test_X, self.train_y, self.val_test_y = train_test_split(self.X, self.y, test_size = val_test_size * 2,
                                                                              random_state=random_state)
        self.val_X, self.test_X, self.val_y, self.test_y = train_test_split(self.val_test_X, self.val_test_y, test_size = 0.5,
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

    def preprocessing(self, val_test_size, random_state):
        self.X = self.pd_data.iloc[::, :-1].values
        self.user_num = max( self.X[::,0] ) + 1
        self.item_num = max( self.X[::, 1] ) + 1
        self.y = self.pd_data.iloc[::, -1].values
        self.train_X, self.val_test_X, self.train_y, self.val_test_y = train_test_split(self.X, self.y, test_size = val_test_size * 2,
                                                                              random_state=random_state)
        self.val_X, self.test_X, self.val_y, self.test_y = train_test_split(self.val_test_X, self.val_test_y, test_size = 0.5,
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
