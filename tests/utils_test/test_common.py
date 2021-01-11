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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
class test_common(unittest.TestCase):
    device_info = "cpu:0"
    def test_set_cpu(self):
        set_device("cpu:0")
        # checks that the current devices being used by tf is a cpu
        assert (len(tf.config.experimental.list_physical_devices()) > 0)
       
    def test_to_snake_case(self):
        temp = to_snake_case("i am a string")
        assert(temp == "i_am_a_string")
        temp = to_snake_case("_i am a private string")
        assert(temp == "private_i_am_a_private_string")
        temp = to_snake_case("IAmStringWithCaps")
        assert(temp == "i_am_string_with_caps")
        temp = to_snake_case("I#am%a&string(with*special+characters")
        assert(temp == "i_am_a_string_with_special_characters")
        
    #Creates a directory and sees if it exists
    def test_create_directory(self):
        assert(os.path.exists("test_dir")==False)
        create_directory("test_dir")
        assert(os.path.exists("test_dir")==True)
    
    #Tests for panda dataframe for 5 possible inputs
    def test_load_dataframe_input(self):
        #Test for panda dataframe
        assert(isinstance(load_dataframe_input(pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})), pd.DataFrame))

        #Test for np_ndarray
        assert(isinstance(load_dataframe_input(np.array( [ 1, 2, 3])), pd.Series))
        
        assert(isinstance(load_dataframe_input(np.array( [[ 1, 2, 3], [ 4, 2, 5]] )), pd.DataFrame))
        #Test for string 
        
        try:
            load_dataframe_input("wrong_file.exe")
        except TypeError:
            assert(True)
        assert(isinstance(load_dataframe_input("test.csv"), pd.DataFrame))
        
    #Sets seed then compares the output to the expected output
    def test_set_seed(self):
        set_seed(10);
        temp = random.random()
        random.seed(10)
        assert(random.random() == temp)
        
        temp = np.random.rand(1);
        np.random.seed(10)
        assert(np.random.rand(1)==temp)
        
        temp = tf.random.uniform([1])
        tf.random.set_seed(10)
        assert(tf.random.uniform([1]) == temp)
        
    #Test save and load pickle
    def test_save_pickle(self):
        save_pickle("test_pickle", { "lion": "yellow", "kitty": "red" })
        assert(os.path.exists("test_pickle") == True)
        
    def test_load_pickle(self):
        save_pickle("test_pickle", { "lion": "yellow", "kitty": "red" })
        temp = load_pickle("test_pickle")
        assert(temp == { "lion": "yellow", "kitty": "red" })