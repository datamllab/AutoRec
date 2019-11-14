from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
import tensorflow as tf

def dataset_shape(dataset):
    return tf.compat.v1.data.get_output_shapes(dataset)

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
