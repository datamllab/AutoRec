import pytest
import os
from autorecsys.utils.common import (
    set_device,
#    dataset_shape,
    to_snake_case,
    create_directory,
    load_dataframe_input,
    set_seed,
    save_pickle,
    load_pickle,
)
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np
import random

@pytest.fixture
def device_info():
    return "cpu:0"

def snake_string():
    return "I am a string"

def snake_string_private():
    return "_I am a private string"

def directory():
    return "test_dir"

def pd_dataframe():
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.Dataframe(data=d)
    return df

def np_ndarray_series():
    arr = np.array( [ 1, 2, 3])
    return arr
    
def np_ndarray_dataframe():
    arr = np.array( [[ 1, 2, 3],
                 [ 4, 2, 5]] )
    return arr


    
class test_common():
    def test_set_cpu():
        set_device(device_info)
        print(device_lib.list_local_devices())
        print(tf.config.experimental.list_physical_devices())
        # checks that the current devices being used by tf is a cpu
        assert (len(tf.config.experimental.list_physical_devices()) > 0)
       
        
    def test_to_snake_case():
        temp = to_snake_case(snake_string)
        print(temp)
        assert(temp == "I_am_a_string")
        temp = to_snake_case(snake_string_private)
        print(temp)
        assert(temp == "private_I_am_a_private_string")
        
    def test_create_directory():
        create_directory(directory)
        print(directory)
        assert(os.path.exists(directory)==True)
    
        
    def test_load_dataframe_input():
        #Test for panda dataframe
        temp = load_dataframe_input(pd_dataframe())
        print(temp)
        assert(isinstance(load_dataframe_input(pd_dataframe()), pd.DataFrame))
        
        
        #Test for np_ndarray
        assert(isinstance(load_dataframe_input(np_ndarray_series()), pd.Series))
        
        assert(isinstance(load_dataframe_input(np_ndarray_dataframe()), pd.DataFrame))
        
        
        #Test for string 
        
        try:
            load_dataframe_input("wring_file.exe")
        except TypeError:
            print("Properly handled wrong file extension")
            assert(True)
        assert(isinstance(load_dataframe_input("test.csv"), pd.DataFrame))
    
    def test_set_seed():
        set_seed(10);
        temp = random.random()
        random.seed(10)
        print(temp)
        assert(random.random() == temp)
        
        temp = np.random.rand(1);
        np.random.seed(10)
        print(temp)
        assert(np.random.rand(1)==temp)
        
        temp = tf.random.uniform([1])
        tf.random.set_seed(10)
        print(temp)
        assert(tf.random.uniform([1]) == temp)
    
    def test_save_pickle(path, obj):
        save_pickle(path, obj)
        load_pickle(path)
        print("Save and load sucessful")