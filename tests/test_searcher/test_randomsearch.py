import os
import pytest
from os.path import join

from autokaggle.base import metric
from autokaggle.base import hyperparameters as hps_module
from autokaggle.base import trial as trial_module
from autokaggle.base.util import set_all_seeds, load_dict_input
from autokaggle.base.pipeline import MLPipeline
from autokaggle.base.oracle import Oracle, Objective
from autokaggle.base.datamanager import TabDataManager

from autokaggle.tuner.randomsearch import RandomSearch

DATA_PATH = '../data/adult'
TRAIN = join(DATA_PATH, 'data', 'train.csv')
TEST = join(DATA_PATH, 'data', 'test.csv')
TEMPLATE = '../config/default_config.yaml'


@pytest.fixture(scope='session')
def template():
    template = load_dict_input(x=TEMPLATE, valid_cond=['pipeline'])
    return template


@pytest.fixture(scope='session')
def dm():
    info = load_dict_input(x=join(DATA_PATH, 'info.yaml'), valid_cond=['id_column', 'label_column',
                                                                       'task_type', 'dataset_type'])
    schema = load_dict_input(x=join(DATA_PATH, 'schema.yaml'), valid_cond=[])
    _dm = TabDataManager(info=info,
                         existed_schema=schema,
                         combine_train_test=True)
    train = _dm.load(TRAIN)
    label_col = info['label_column']
    y_tra = train[label_col]
    x_tra = train.drop(columns=[label_col])
    del train
    _dm.get_information_from_train_data(x_tra, y_tra)
    return _dm


@pytest.fixture(scope='session')
def tmp_dir(tmpdir_factory):
    set_all_seeds(seed=0)
    return tmpdir_factory.mktemp('integration_test', numbered=True)


def test_randomsearch(dm, template, tmp_dir):
    pipe = MLPipeline.from_dict(dm=dm, template=template)
    hps = pipe.parse_hparams_dict_to_hyperparameters(pipe._tunable_parameters)
    tuner = RandomSearch(model=pipe, objective='logloss', max_trials=50)
    tuner.search()
