import pytest
from autorecsys.utils.common import set_device
import tensorflow as tf
from tensorflow.python.client import device_lib

@pytest.fixture
def device_info():
    return "cpu:0"

def test_set_cpu(device_info):
    set_device(device_info)
    print(device_lib.list_local_devices())
    print(tf.config.experimental.list_physical_devices())
    # checks that the current devices being used by tf is a cpu
    assert (len(tf.config.experimental.list_physical_devices()) > 0)
