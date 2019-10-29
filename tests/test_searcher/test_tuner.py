import pytest
from os.path import join

from autorecsys.base.util import load_dict_input
from autorecsys.pipeline.recommender import Recommender

from autorecsys.searcher.randomsearch import RandomSearch

DATA_PATH = '../data/adult'
TRAIN = join(DATA_PATH, 'data', 'train.csv')
TEST = join(DATA_PATH, 'data', 'test.csv')
TEMPLATE = '../config/default_config.yaml'


@pytest.fixture(scope='session')
def template():
    template = load_dict_input(x=TEMPLATE, valid_cond=['pipeline'])
    return template


@pytest.fixture(scope='session')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test', numbered=True)


def test_randomsearch(dm, template, tmp_dir):
    pipe = Recommender(dm)
    hps = pipe.parse_hparams_dict_to_hyperparameters(pipe._tunable_parameters)
    tuner = RandomSearch(model=pipe, objective='logloss', max_trials=50)
    tuner.search()
