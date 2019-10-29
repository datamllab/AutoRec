from __future__ import absolute_import, division, print_function, unicode_literals

from autorecsys.utils.config import load_config, extract_tunable_hps
from autorecsys.utils.common import set_device
from autorecsys.utils import load_dataset
from autorecsys.pipeline.recommender import Recommender
from autorecsys.trainer import train
from autorecsys.searcher.randomsearch import RandomSearch


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
        self.model, self.avg_loss = train(self.model, self.data, self.job_config)

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
        self.searcher = RandomSearch(config=self.search_config,
                                     objective=self.search_config["SearchOption"]["objective"],
                                     max_trials=self.search_config["SearchOption"]["max_trials"],
                                     hyperparameters=self.hps,
                                     dataset=self.data,
                                     overwrite=True)
        self.model = None

    def search(self):
        # train model
        self.searcher.search()
        return self.searcher.results_summary()

    def final_eval(self, test_file):
        # TODO:
        pass
