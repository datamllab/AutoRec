import os
import pytest
import tensorflow as tf
from autorecsys.pipeline.interactor import ConcatenateInteraction, ElementwiseAddInteraction
from autorecsys.searcher.core import hyperparameters as hp_module

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warning for running TF with CPU


@pytest.fixture
def inputs():
    v1 = tf.constant([1, 2, 3])
    v2 = tf.constant([4, 5, 6])
    return [v1, v2]


def test_concatenate(inputs):
    """
    Test class ConcatenateInteraction in interactor.py
    """
    sol = tf.constant([1, 2, 3, 4, 5, 6])  # Arrange
    hp = hp_module.HyperParameters()
    interactor = ConcatenateInteraction()

    ans = interactor.build(hp, inputs)  # Act

    assert all(tf.equal(ans, sol))  # Assert


def test_elementwise_add(inputs):
    """
    Test class ElementwiseAddInteraction in interactor.py
    """
    sol = tf.constant([5, 7, 9])  # Arrange
    hp = hp_module.HyperParameters()
    interactor = ElementwiseAddInteraction()

    ans = interactor.build(hp, inputs)  # Act

    assert all(tf.equal(ans, sol))  # Assert
