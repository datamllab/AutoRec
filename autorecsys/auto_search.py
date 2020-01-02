from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import tempfile
import tensorflow as tf
from copy import deepcopy
from sklearn.utils import safe_indexing
from sklearn.model_selection import StratifiedShuffleSplit

from autorecsys.utils.common import to_snake_case, create_directory, preprocess_xy, \
    get_available_components, load_dataframe_input
from autorecsys.pipeline.recommender import CFRecommender, CTRRecommender
from autorecsys.searcher.core.tuner import PipeTuner, METRIC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Search(object):
    def __init__(self, task='cf', name=None, tuner=None, tuner_params=None, directory='.', overwrite=True, **kwargs):
        self.task = task
        if self.task == 'cf':
            self.pipe = CFRecommender(**kwargs)
        elif self.task == 'ctr':
            self.pipe = CTRRecommender(**kwargs)
        else:
            raise ValueError(
                'Currently we only support "cf" and "ctr" tasks.')
        self.tuner = tuner
        self.tuner_params = tuner_params
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(tf.keras.backend.get_uid(prefix))
            name = to_snake_case(name)
        self.name = name
        directory = directory or tempfile.gettempdir()
        self.dir = os.path.join(directory, self.name)

        self.overwrite = overwrite
        create_directory(self.dir, remove_existing=overwrite)
        self.logger = logging.getLogger(self.name)
        self.logger.info('Project directory: {}'.format(self.dir))
        self.best_keras_graph = None
        self.best_model = None
        self.need_fully_train = False

    def search(self, x=None, y=None, x_val=None, y_val=None, objective=None, batch_size=None):
        # overwrite the objective
        self.objective = objective or 'mse'
        tuner = self._build_tuner(self.tuner, self.tuner_params)
        # show the search space
        tuner.search_space_summary()

        # TODO search on a small piece of train data, currently it uses whole train data
        tuner.search(x=x, y=y, x_val=x_val, y_val=y_val, batch_size=batch_size)
        tuner.results_summary()
        best_pipe_lists = tuner.get_best_models(1)
        # len(best_pipe_lists) == 0 means that this pipeline does not have tunable parameters
        self.best_keras_graph, self.best_model = best_pipe_lists[0]
        self.best_keras_graph.save(tuner.best_keras_graph_path)
        self.logger.info('retrain the best pipeline using whole data set')

        self.best_model.save_weights(tuner.best_model_path)
        return self.best_model

    def _build_tuner(self, tuner, tuner_params):
        available_tuners = get_available_components(package=f'{__package__}.searcher',
                                                    directory=os.path.join(os.path.split(__file__)[0], 'searcher'),
                                                    base_class=PipeTuner)
        if tuner is None:
            tuner_cls = available_tuners.items()[0][0]
        else:
            try:
                tuner_cls = available_tuners[tuner]
            except Exception:
                raise Exception(f'currently do not support {tuner}, available tuners: {list(available_tuners.keys())}')
        hps = self.pipe.get_hyperparameters()
        tuner = tuner_cls(hypergraph=self.pipe,
                          objective=self.objective,
                          hyperparameters=hps,
                          directory=self.dir,
                          **tuner_params)
        return tuner

    def predict(self, x):
        x = load_dataframe_input(x) if self.task == "cf" else x
        return self.best_model.predict(x)

    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        score_func = METRIC[self.objective.split('_')[-1]]
        y_true = load_dataframe_input(y_true)
        y_true = y_true.values.reshape(-1, 1)
        self.logger.info(f'evaluate prediction results using {self.objective}')
        return score_func(y_true, y_pred)
