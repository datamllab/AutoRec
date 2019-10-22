from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.utils import load_config, extract_tunable_hps, set_device
from autorecsys.data import load_dataset
from autorecsys.pipeline.recommender import Recommender
from autorecsys.trainer import train
from autorecsys.searcher.random_search import RandomSearch


class Job(object):
    def __init__(self, job_config):
        self.job_config = load_config(job_config)
        # set device
        set_device(self.job_config["TrainOption"]["device"])
        # load data
        self.data = load_dataset(self.job_config)
        # build model
        self.model = Recommender(self.job_config["ModelOption"])

    def run(self):
        # train model
        self.model = train(self.model, self.data)

    def eval(self, test_file):
        # TODO:
        pass


class AutoSearch(object):
    def __init__(self, search_config):
        self.search_config = load_config(search_config)
        self.hps = extract_tunable_hps(self.search_config["ModelOption"])
        # set device
        set_device(self.search_config["TrainOption"]["device"])
        # load data
        self.data = load_dataset(self.search_config)
        self.searcher = RandomSearch(model=Recommender,
                                     objective='mse',
                                     max_trials=3,
                                     hyperparameters=self.hps,
                                     dataset=self.data)
        self.model = None

    def search(self):
        # train model
        self.model = self.searcher.search()
        return self.model

    def final_eval(self, test_file):
        # TODO:
        pass
