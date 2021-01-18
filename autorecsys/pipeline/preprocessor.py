# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math


class BasePreprocessor(metaclass=ABCMeta):
    """ Preprocess data into Pandas DataFrame format.

    # Arguments
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.

    # Attributes
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        data_df (DataFrame): Data loaded in Pandas DataFrame format and contains only relevant columns.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dict (dict): Map string categorical column names to dictionary which maps categories to indices.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.
    """

    @abstractmethod
    def __init__(self,
                 non_csv_path=None,
                 csv_path=None,
                 header=None,
                 columns=None,
                 delimiter=None,
                 filler=None,
                 dtype_dict=None,
                 ignored_columns=None,
                 target_column=None,
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=None,
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=None,
                 validate_percentage=None,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        super().__init__()
        # Dataset load attributes.
        self.non_csv_path = non_csv_path
        self.csv_path = csv_path
        self.header = header
        self.columns = columns
        self.delimiter = delimiter
        self.filler = filler
        self.dtype_dict = dtype_dict
        self.ignored_columns = ignored_columns
        self.data_df = None

        # Dataset access attributes.
        self.target_column = target_column
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        # Dataset transformation attributes.
        self.categorical_filter = categorical_filter
        self.fit_dict = None
        self.fit_dictionary_path = fit_dictionary_path
        self.transform_path = transform_path

        # Dataset split attributes.
        self.test_percentage = test_percentage
        self.validate_percentage = validate_percentage
        self.train_path = train_path
        self.validate_path = validate_path
        self.test_path = test_path

    def format_dataset(self):
        """ (Optional) Convert dataset into CSV format.

        # Note
            User should implement this function to convert non-CSV dataset into CSV format.
        """
        raise NotImplementedError

    def load_dataset(self):  # pragma: no cover
        """ Load CSV data as a Pandas DataFrame object.
        """
        self.data_df = pd.read_csv(self.csv_path, sep=self.delimiter, header=self.header, names=self.columns, dtype=self.dtype_dict)
        self.data_df.drop(columns=self.ignored_columns, inplace=True)
        self.data_df.fillna(self.filler, inplace=True)

    def transform_categorical(self):
        """ Transform categorical data.

        # Note
            Produce fit dictionary for categorical data and transform categorical data using fit dictionary.
        """
        # Step 1: Count categorical occurrences for each column.
        self.fit_dict = {col: self.data_df[col].value_counts().to_dict() for col in self.categorical_columns}

        # Step 2: Reindex categories for each column (create fit dictionary)
        for col, count_dict in self.fit_dict.items():
            index = 0.0  # float meets TensorFlow type requirement
            categories = list()
            for category, count in count_dict.items():
                if count > self.categorical_filter:
                    self.fit_dict[col][category] = index
                    index += 1
                else:
                    categories.append(category)
            for category in categories:
                self.fit_dict[col][category] = index

        # Step 3: Transform categorical data (apply fit dictionary)
        for col in self.categorical_columns:
            self.data_df[col] = self.data_df[col].map(self.fit_dict[col])  # set class attribute pd_data

    def transform_numerical(self):
        """ Transform numerical data using supported data transformation functions.
        """
        # Step 1: Define transformation method.
        def scale_by_log(num):
            """ Scale numerical data by log transformation.
            """
            return math.log(float(num)) ** 2 if num > 2 else num

        # Step 2: Transform numerical data (apply transformation method)
        for col in self.numerical_columns:
            self.data_df[col] = self.data_df[col].map(scale_by_log)

    def get_hash_size(self):
        """ Get the hash sizes of categorical columns.

        # Returns
            List of integer numbers of categories in each categorical data columns.
        """
        return [len(self.fit_dict[col]) for col in self.categorical_columns]

    def get_x(self):
        """ Get the training data columns.

        # Returns
            DataFrame training data columns.
        """
        return self.data_df.drop(columns=self.target_column, inplace=False)

    def get_x_numerical(self, x):
        """ Get the numerical columns from the input data columns.

        # Arguments
            x (DataFrame): The input data columns.

        # Returns
            ndarray numerical columns in the input data columns.
        """
        return x[self.numerical_columns].values

    def get_x_categorical(self, x):
        """ Get the categorical columns from the input data columns.

        # Arguments
            x (DataFrame): The input data columns.

        # Returns
            ndarray categorical columns in the input data columns.
        """
        return x[self.categorical_columns].values

    def get_y(self):
        """ Get the output column.

        # Returns
            ndarray output column.
        """
        return self.data_df[self.target_column].values

    def get_numerical_count(self):
        """ Get the number of numerical columns.

        # Returns
            Integer number of numerical columns.
        """
        return len(self.numerical_columns)

    def get_categorical_count(self):
        """ Get the number of categorical columns.

        # Returns
            Integer number of categorical columns.
        """
        return len(self.categorical_columns)

    def split_data(self, x, y, test_percentage):
        """ Split data into the train, validation, and test sets.

        # Arguments
            x (ndarray): (M, N) matrix associated with the numerical and categorical data, where M is the number of rows
                and N is the number of numerical and categorical columns.
            y (ndarray): (M, 1) matrix associated with the label data, where M is the number of rows.
            test_percentage (float): Percentage of test set.

        # Returns
            4-tuple of ndarray training input data, testing input data, training output data, and testing output data.
        """
        test_size = math.ceil(test_percentage * len(self.data_df.index))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y)

        return x_train, x_test, y_train, y_test

    @abstractmethod
    def preprocess(self):
        """ Apply all preprocess steps.

        # Note
            User is responsible for calling the needed data preprocessing functions from here.

        # Returns
            6-tuple of ndarray training input data, training output data, validation input data, validation output data,
                testing input data, and testing output data.
        """
        raise NotImplementedError


class AvazuPreprocessor(BasePreprocessor):
    """ Preprocess the Avazu dataset for click-through rate (CTR) prediction.

    # Note:
        To obtain the Avazu dataset, visit: https://www.kaggle.com/c/avazu-ctr-prediction
        The Avazu dataset has 24 data columns: [0] is ignored, [1] is label, and [2-23] are categorical.

    # Arguments
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.
    """

    def __init__(self,
                 non_csv_path=None,
                 csv_path='./example_datasets/avazu/train-10k',
                 header=0,
                 columns=None,
                 delimiter=',',
                 filler=0.0,
                 dtype_dict=None,  # inferred in load_data()
                 ignored_columns=None,
                 target_column='click',
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=3,
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        if columns is None:
            columns = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id',
                       'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type',
                       'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
        if dtype_dict is None:
            dtype_dict = {}
        if ignored_columns is None:
            ignored_columns = columns[0]
        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = columns[2:]

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

    def preprocess(self):
        """ Apply all preprocessing steps to the Avazu dataset.

        # Returns
            6-tuple of ndarray training input data, training output data, validation input data, validation output data,
                testing input data, and testing output data.
        """
        # Step 1: Load the Avazu dataset.
        self.load_dataset()

        # Step 2: Transform categorical data.
        self.transform_categorical()

        # Step 3: Obtain LHS (X) and RHS (y) of the equation.
        x = self.get_x()
        y = self.get_y()

        # Step 4: Split data for training, validation, and testing.
        x_train, x_test, y_train, y_test = self.split_data(x, y, self.test_percentage)
        x_train, x_validate, y_train, y_validate = self.split_data(x_train, y_train, self.validate_percentage)

        return x_train, y_train, x_validate, y_validate, x_test, y_test


class CriteoPreprocessor(BasePreprocessor):
    """ Preprocess the Criteo dataset for click-through rate (CTR) prediction.

    # Note
        To obtain the Criteo dataset, visit: https://www.kaggle.com/c/criteo-display-ad-challenge/
        The Criteo dataset has 40 data columns: [0] is label, [1-13] are numerical, and [14-39] are categorical.

    # Arguments
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.
    """

    def __init__(self,
                 non_csv_path=None,
                 csv_path='./example_datasets/criteo/train-10k.txt',
                 header=None,  # inferred in load_data()
                 columns=None,
                 delimiter='\t',
                 filler=0.0,
                 dtype_dict=None,  # inferred in load_data()
                 ignored_columns=None,
                 target_column=0,
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=3,
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        if columns is None:
            columns = range(40)
        if dtype_dict is None:
            dtype_dict = {}
        if ignored_columns is None:
            ignored_columns = []
        if numerical_columns is None:
            numerical_columns = range(1, 14)
        if categorical_columns is None:
            categorical_columns = range(14, 40)

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

    def preprocess(self):
        """ Apply all preprocessing steps to the Criteo dataset.

        # Returns
            6-tuple of ndarray training input data, training output data, validation input data, validation output data,
                testing input data, and testing output data.
        """
        # Step 1: Load data for fit and transform categorical data.
        self.load_dataset()

        # Step 2: Transform categorical data.
        self.transform_categorical()

        # Step 3: Transform numerical data.
        self.transform_numerical()

        # Step 4: Obtain LHS (X) and RHS (y) of the equation.
        x = self.get_x()
        y = self.get_y()

        # Step 5: Split data for training, validation, and testing.
        x_train, x_test, y_train, y_test = self.split_data(x, y, self.test_percentage)
        x_train, x_validate, y_train, y_validate = self.split_data(x_train, y_train, self.validate_percentage)

        return x_train, y_train, x_validate, y_validate, x_test, y_test


class NetflixPrizePreprocessor(BasePreprocessor):
    """ Preprocess the Netflix dataset for rating prediction.

    # Note
        To obtain the Netflix dataset, visit: https://www.kaggle.com/netflix-inc/netflix-prize-data
        The Netflix dataset has 4 data columns: MovieID, CustomerID, Rating, and Date.

    # Arguments
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.
    """

    def __init__(self,
                 non_csv_path='./example_datasets/netflix/combined_data_1-10k.txt',
                 csv_path='./example_datasets/netflix/combined_data_1-10k.csv',
                 header=None,  # inferred in load_data()
                 columns=None,
                 delimiter=',',
                 filler=0.0,
                 dtype_dict=None,
                 ignored_columns=None,
                 target_column='Rating',
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=0,  # no grouping, simply renumber
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        if columns is None:
            columns = ['MovieID', 'CustomerID', 'Rating', 'Date']
        if dtype_dict is None:
            dtype_dict = {'MovieID': np.float32, 'CustomerID': np.float32, 'Rating': np.float32, 'Date': np.str}
        if ignored_columns is None:
            ignored_columns = columns[3]
        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = columns[:2]

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

    def format_dataset(self):
        """ Convert the Netflix Prize dataset into CSV format and save it as a new file.

        # Note:
            This is an example showing the function which converts dataset into the CSV format as provided by user.
        """
        with open(self.non_csv_path, 'r') as rf, open(self.csv_path, 'w') as wf:
            for line in rf:
                if ':' in line:
                    movie = line.strip(":\n")
                else:
                    wf.write(movie + ',' + line)

    def preprocess(self):
        """ Apply all preprocessing steps to the Netflix Prize dataset.

        # Returns
            6-tuple of ndarray training input data, training output data, validation input data, validation output data,
                testing input data, and testing output data.
        """
        # Step 1: Convert Netflix dataset to CSV format.
        self.format_dataset()

        # Step 2: Load data for fit and transform categorical data.
        self.load_dataset()

        # Step 3: Transform categorical data.
        self.transform_categorical()

        # Step 4: Obtain LHS (X) and RHS (y) of the equation.
        x = self.get_x()
        y = self.get_y()

        # Step 5: Split data for training, validation, and testing.
        x_train, x_test, y_train, y_test = self.split_data(x, y, self.test_percentage)
        x_train, x_validate, y_train, y_validate = self.split_data(x_train, y_train, self.validate_percentage)

        return x_train, y_train, x_validate, y_validate, x_test, y_test


class MovielensPreprocessor(BasePreprocessor):
    """ Preprocess the Movielens 1M dataset for rating prediction.

    # Note
        To obtain the Movielens 1M dataset, visit: https://grouplens.org/datasets/movielens/
        The Movielens 1M dataset has 4 data columns: UserID, MovieID, Rating, and Timestamp.

    # Arguments
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.
    """

    def __init__(self,
                 non_csv_path=None,
                 csv_path='./example_datasets/movielens/ratings-10k.dat',
                 header=None,  # inferred in load_data()
                 columns=None,
                 delimiter='::',
                 filler=0.0,
                 dtype_dict=None,
                 ignored_columns=None,
                 target_column='Rating',
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=0,  # no grouping, simply renumber
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        if columns is None:
            columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        if dtype_dict is None:
            dtype_dict = {'UserID': np.int32, 'MovieID': np.int32, 'Rating': np.int32, 'Timestamp': np.int32}
        if ignored_columns is None:
            ignored_columns = columns[3]
        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = columns[:2]

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

    def preprocess(self):
        """ Apply all preprocessing steps to the Movielens 1M dataset.

        # Returns
            6-tuple of ndarray training input data, training output data, validation input data, validation output data,
                testing input data, and testing output data.
        """
        # Step 1: Load data for fit and transform categorical data.
        self.load_dataset()

        # Step 2: Transform categorical data.
        self.transform_categorical()

        # Step 3: Obtain LHS (X) and RHS (y) of the equation.
        x = self.get_x()
        y = self.get_y()

        # Step 4: Split data for training, validation, and testing.
        x_train, x_test, y_train, y_test = self.split_data(x, y, self.test_percentage)
        x_train, x_validate, y_train, y_validate = self.split_data(x_train, y_train, self.validate_percentage)

        return x_train, y_train, x_validate, y_validate, x_test, y_test
