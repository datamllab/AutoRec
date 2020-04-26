from __future__ import absolute_import, division, print_function, unicode_literals

import re
import os
import shutil
import pandas as pd
import numpy as np
import inspect
import pkgutil
from collections import OrderedDict
import importlib
import tensorflow as tf
import random
from tensorflow.python.util import nest
import pickle


def dataset_shape(dataset):
    return tf.compat.v1.data.get_output_shapes(dataset)


def to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


def create_directory(path, remove_existing=False):
    # Create the directory if it doesn't exist.
    if not os.path.exists(path):
        os.mkdir(path)
    # If it does exist, and remove_existing is specified,
    # the directory will be removed and recreated.
    elif remove_existing:
        shutil.rmtree(path)
        os.mkdir(path)


def set_device(device_name):
    if device_name[0:3] == "cpu":
        cpus = tf.config.experimental.list_physical_devices('CPU')
        print("Available CPUs: {}".format(cpus))
        assert len(cpus) > 0, "Not enough CPU hardware devices available"
        cpu_idx = int(device_name[-1])
        tf.config.experimental.set_visible_devices(cpus[cpu_idx], 'CPU')
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Available GPUs: {}".format(gpus))
        assert len(gpus) > 0, "Not enough GPU hardware devices available"
        gpu_idx = int(device_name[-1])
        tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')


def load_dataframe_input(x):
    if x is None:
        return None
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        res = x
    elif isinstance(x, np.ndarray):
        res = pd.Series(x) if len(x.shape) == 1 else pd.DataFrame(x)
    elif isinstance(x, str):
        if not x.endswith('.csv'):
            raise TypeError(f'ONLY accept path to the local csv files')
        res = pd.read_csv(x)
    else:
        raise TypeError(f"cannot load {type(x)} into pandas dataframe")
    # make sure the returned dataframe's col name data type is string
    if isinstance(res, pd.DataFrame):
        res.columns = res.columns.astype('str')
    return res


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_pickle(path, obj):
    # TODO: create directory if it does not exist
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
