from __future__ import absolute_import, division, print_function, unicode_literals

import re
import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import pickle
import string


def dataset_shape(dataset):
    """ Get the shape of the dataset.

    Args:
        dataset (tf.data.Dataset or Tf.data.Iterator): A TensorFlow Dataset or Iterator.

    Returns:
        A nested structure of tf.TensorShape object matching the structure of the dataset / iterator elements and
            specifying the shape of the individual components.
    """
    return tf.compat.v1.data.get_output_shapes(dataset)


def to_snake_case(name):
    """ Convert the given class name to snake case.

    # Arguments
        name (str): The name of the class.

    # Returns
        String name of the class in snake case.
    """
    insecure = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    insecure = re.sub('([a-z0-9])([A-Z])', r'\1_\2', insecure).lower()
    for p in string.punctuation:
        insecure = insecure.replace(p, "_")

    if insecure[0] != '_':
        return insecure
    # A private class (starts with "_") is not secure for creating scopes and is thus prefixed w/ "private".
    return 'private' + insecure


def create_directory(path, remove_existing=False):
    """ Create the designated directory.

    # Arguments
        path (str): Path to create the directory.
        remove_existing (bool): Whether to remove the directory if it already exists.
    """
    # Create the directory if it doesn't exist.
    if not os.path.exists(path):
        os.mkdir(path)
    # Remove the preexisting directory if allowed.
    elif remove_existing:
        shutil.rmtree(path)
        os.mkdir(path)


def set_device(device_name):
    """ Set the computational devices used to run models.

    # Arguments
        device_name (str): Name of the CPU or GPU.
    """
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
    """ Load the input object as a DataFrame or a Series.

    # Note
        Cover the following classes: None, DataFrame, Series, ndarray, and str.

    # Arguments
        x (object): The object to be loaded as a DataFrame or Series.

    # Returns
        The loaded DataFrame or Series.
    """
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

    # Ensure the type of column names is string
    if isinstance(res, pd.DataFrame):
        res.columns = res.columns.astype('str')
    return res


def set_seed(seed=42):
    """ Set the seed for randomization functions.

    # Note
        Cover the following libraries: Python, Numpy, and TensorFlow

    # Arguments
        seed (float): The seed number used to create fixed randomization.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_pickle(path, obj):
    """ Save the input object to the designated path.

    # Arguments
        path (str): Designated path to save the object.
        obj (object): The object to be saved.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """ Load the object file from the designated path.

    # Arguments
        path: Designated path to load the object.

    Returns:
        The loaded object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
