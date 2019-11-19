from autorecsys.pipeline.interactor import *
from autorecsys.searcher.core import hyperparameters as hp_module


def assert_tensor_operation(ans, sol):
    result = tf.equal(ans, sol)
    assert False not in result


def test_concatenate():
    """
    Test class ConcatenateInteraction in interactor.py
    """
    v1 = tf.constant([1, 2, 3])  # Arrange
    v2 = tf.constant([4, 5, 6])
    sol = tf.constant([1, 2, 3, 4, 5, 6])

    inputs = [v1, v2]
    hp = hp_module.HyperParameters()
    interactor = ConcatenateInteraction()

    ans = interactor.build(hp, inputs)  # Act

    assert_tensor_operation(ans, sol)  # Assert


def test_elementwise_add():
    """
    Test class ElementwiseAddInteraction in interactor.py
    """
    v1 = tf.constant([1, 2, 3])  # Arrange
    v2 = tf.constant([4, 5, 6])
    sol = tf.constant([5, 7, 9])

    inputs = [v1, v2]
    hp = hp_module.HyperParameters()
    interactor = ElementwiseAddInteraction()

    ans = interactor.build(hp, inputs)  # Act

    assert_tensor_operation(ans, sol)  # Assert


if __name__ == "__main__":

    test_concatenate()
    test_elementwise_add()
