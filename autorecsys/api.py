from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.utils import load_config
from autorecsys.data import load_dataset
from autorecsys.pipeline.recommender import Recommender
from autorecsys.trainer import train


class Job(object):
    def __init__(self, job_config):
        self.job_config = load_config(job_config)
        # set device
        self._set_device()
        # load data
        self.data = load_dataset(self.job_config)
        # build model
        self.model = Recommender(self.job_config["ModelOption"])

    def _set_device(self):
        if self.job_config["TrainOption"]["device"] == "cpu":
            pass
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print("Available GPUs: {}".format(gpus))
            assert len(gpus) > 0, "Not enough GPU hardware devices available"
            gpu_idx = int(self.job_config["TrainOption"]["device"][-1])
            tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')

    def run(self):
        # train model
        self.model = train(self.model, self.data)

    def eval(self, test_file):
        # TODO:
        pass
