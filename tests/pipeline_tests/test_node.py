import copy

import numpy as np
import pytest
import tensorflow as tf

from autorecsys.pipeline import node


def test_input_type_error():
    x = 'unknown'
    input_node = node.Input()
    with pytest.raises(TypeError) as info:
        input_node._check(x)
        x = input_node.transform(x)
    assert 'Expect the data to Input to be numpy' in str(info.value)


def test_input_numerical():
    x = np.array([[['unknown']]])
    input_node = node.Input()
    with pytest.raises(TypeError) as info:
        input_node._check(x)
        x = input_node.transform(x)
    assert 'Expect the data to Input to be numerical' in str(info.value)


