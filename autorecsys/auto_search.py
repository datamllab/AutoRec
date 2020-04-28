from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import tempfile
import tensorflow as tf

from autorecsys.utils.common import to_snake_case, create_directory,  load_dataframe_input
from autorecsys.searcher.tuners.tuner import METRIC, PipeTuner
from autorecsys.searcher import tuners
from autorecsys.recommender import CTRRecommender, RPRecommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Search(object):
    """ A search object to search on a Recommender HyperModel (CTRRecommender/RPRecommender) 
    defined by inputs and outputs.

    ``Search`` combines a Recommender and a Tuner to tune the Recommender. The user can 
    use ``search()`` to perform search, and use a similar way to a Keras model to adopt 
    the best discovered model as it also has `fit()`/`predict()`/`evaluate()` methods.
    The user should input a Recommender HyperModel (CTRRecommender/RPRecommender) and a 
    selected tuning method to initial the ``Search`` object and input the dataset when 
    calling the ``search`` method to discover the best architecture.  
    ```
    # Arguments
        model: A Recommender HyperModel (CTRRecommender/RPRecommender).
        name: String. The name of the project, which is used for saving and loading purposes.
        tuner: String. The name of the tuner. It should be one of 'greedy', 'bayesian' or 
            'random'. Default to be 'random'.


        tuner_params: Dict. The hyperparameters of the tuner. The commons ones are:
                 'max_trials': Int. Specify the number of search epochs.
                 'overwrite': Boolean. Whether we want to ovewrite an existing 
                    tuner or not.

        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            project in the current directory, i.e., ``directory/name``.
        overwrite: Boolean. Defaults to `True`. Whether we want to ovewrite an existing 
            project with the name defined as ``directory/name`` or not.
    """
    def __init__(self, model=None, name=None, tuner='random', tuner_params=None, directory='.', overwrite=True):
        self.pipe = model
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

    def search(self, x=None, y=None, x_val=None, y_val=None, objective='mse', **fit_kwargs):
        """Search the best deep recommendation model.

        # Arguments
            x: numpy array. Training features.
            y: numpy array. Training targets.
            x_val: numpy array. Validation features.
            y_val: numpy array. Validation features.
            objective: String. Name of model metric to minimize or maximize, 
                e.g. 'val_BinaryCrossentropy'. Defaults to 'mse'.
            **fit_kwargs: Any arguments supported by the fit method of a Keras model such as: 
                ``batch_size``, ``epochs``, ``callbacks``.
        """

        # overwrite the objective
        self.objective = objective
        tuner = self._build_tuner(self.tuner, self.tuner_params)

        # TODO search on a small piece of train data, currently it uses whole train data
        tuner.search(x=x, y=y, x_val=x_val, y_val=y_val, **fit_kwargs)
        # show the search space
        tuner.search_space_summary()
        # show the search results
        tuner.results_summary()
        best_pipe_lists = tuner.get_best_models(1)
        # len(best_pipe_lists) == 0 means that this pipeline does not have tunable parameters
        self.best_model = best_pipe_lists[0]
        return self.best_model

    def _build_tuner(self, tuner, tuner_params):
        """Build a tuner based on its name and hyperparameters.

        # Arguments
            tuner: String. The name of the tuner. It should be one of 'greedy', 'bayesian' or 
                'random'. Default to be 'random'.

            tuner_params: Dict. The hyperparameters of the tuner. The commons ones are:
                 'max_trials': Int. Specify the number of search epochs.
                 'overwrite': Boolean. Whether we want to ovewrite an existing 
                    tuner or not. 
        """
        tuner_cls = tuners.get_tuner_class( tuner )
        hps = self.pipe.get_hyperparameters()
        tuner = tuner_cls(hypergraph=self.pipe,
                          objective=self.objective,
                          hyperparameters=hps,
                          directory=self.dir,
                          **tuner_params)
        return tuner

    def predict(self, x):
        """Use the best searched model to conduct prediction on the dataset x.

        # Arguments
            x: numpy array / data frame / string path of a csv file. 
                Features used to do the prediction.
        """
        if isinstance (self.pipe, RPRecommender):
            x = load_dataframe_input(x)
        return self.best_model.predict(x)

    def evaluate(self, x, y_true):
        """Evaluate the best searched model.

        # Arguments
            x: numpy array / data frame / string path of a csv file. 
                Features used to do the prediction.
            y_true: numpy array / data frame / string path of a csv file. 
                Ground-truth labels.
        """
        y_pred = self.predict(x)
        score_func = METRIC[self.objective.split('_')[-1]]
        y_true = load_dataframe_input(y_true)
        y_true = y_true.values.reshape(-1, 1)
        self.logger.info(f'evaluate prediction results using {self.objective}')
        return score_func(y_true, y_pred)
