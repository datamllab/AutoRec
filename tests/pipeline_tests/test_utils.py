from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import pytest
import unittest

import numpy as np
import tensorflow as tf
from autorecsys.pipeline.utils import Bias

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU


logger = logging.getLogger(__name__)

class TestBias(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()  # change to pytest-provided temporary directory
        tmpdir.join("test_utils.ini").write("# testdata")

    def setUp(self):
        super(TestBias, self).setUp()
        self.inputs = tf.constant([ [1, 2, 3], [4, 5, 6] ], dtype="float32")
        self.test_units = 4
    
    def test_Bias(self):
        bias = Bias(units=self.test_units)
        assert bias.bias.shape == (self.test_units,)

    def test_call(self):
        """
        Test Bias.call()
        """
        bias = Bias(self.inputs.shape[-1])  # Pass shape of input as units argument
        ans = bias(self.inputs)
        tf.assert_equal(self.inputs, ans)  # Assert tensor is equal since bias layer adds zeroes
