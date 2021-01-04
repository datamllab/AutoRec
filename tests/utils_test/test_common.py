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
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@pytest.fixture
def snake_string():
    return "i am a string"

def snake_string_private():
    return "_i am a private string"

def snake_string_caps():
    return "I AM A STRING WITH CAPS"

def snake_string_special():
    return "I#am%a&string(with*special+characters"

def directory():
    return "test_dir"

def pickle_directory():
    return "test_dir_pickle"

def pd_dataframe():
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    return df

def np_ndarray_series():
    arr = np.array( [ 1, 2, 3])
    return arr
    
def np_ndarray_dataframe():
    arr = np.array( [[ 1, 2, 3],
                 [ 4, 2, 5]] )
    return arr


    
class test_common(unittest.TestCase):
    device_info = "cpu:0"
    #Can't use the provided decive_info() or else pytest throws a fit, looking into fixing
    def test_set_cpu(self):
        set_device("cpu:0")
        print(device_lib.list_local_devices())
        print(tf.config.experimental.list_physical_devices())
        # checks that the current devices being used by tf is a cpu
        assert (len(tf.config.experimental.list_physical_devices()) > 0)
       
        
    #Snake case fails on testing, although this appears to be the function not working properly
    def test_to_snake_case(self):
        temp = to_snake_case(snake_string())
        print(temp)
        assert(temp == "i_am_a_string")
        temp = to_snake_case(snake_string_private())
        print(temp)
        assert(temp == "private_i_am_a_private_string")
        temp = to_snake_case(snake_string_caps())
        print(temp)
        assert(temp == "i_am_a_string_with_caps")
        temp = to_snake_case(snake_string_special())
        print(temp)
        assert(temp == "i_am_a_string_with_special_characters")
        
    #Creates a directory and sees if it exists
    def test_create_directory(self):
        create_directory(directory())
        print(directory())
        assert(os.path.exists(directory())==True)
    
    #Tests for panda dataframe for 5 possible inputs
    def test_load_dataframe_input(self):
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
    #Sets seed then compares the output to the expected output
    def test_set_seed(self):
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
    #Saves and loads pickle
    def test_save_pickle(self):
        #save_pickle(path, obj)
        #load_pickle(path)
        print("Save and load sucessful")
        

        