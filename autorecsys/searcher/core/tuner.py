# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"Tuner base class."

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
import logging

from autorecsys.utils import display
from autorecsys.utils.common import create_directory
from autorecsys.utils.config import set_tunable_hps
from autorecsys.searcher.core import trial as trial_module
from autorecsys.searcher.core import oracle as oracle_module
from autorecsys.pipeline.recommender import Recommender

from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

# TODO: Add more extensive display.

LOGGER = logging.getLogger(__name__)
METRIC = {'auc': roc_auc_score, 'log_loss': log_loss, 'mse': mean_squared_error}


class Display(object):

    @staticmethod
    def on_trial_begin(trial):
        display.section('New model')
        trial.summary()

    @staticmethod
    def on_trial_end(trial):
        display.section('Trial complete')
        trial.summary()


class BaseTuner(trial_module.Stateful):
    """Tuner base class.
    May be subclassed to create new tuners, including for non-Keras models.
    Args:
        oracle: Instance of Oracle class.
        directory: String. Path to the working directory (relative).
        project_name: Name to use as prefix for files saved
            by this Tuner.
        tuner_id: Optional. Used only with multi-worker DistributionStrategies.
        overwrite: Bool, default `False`. If `False`, reloads an existing project
            of the same name if one is found. Otherwise, overwrites the project.
    """

    def __init__(self,
                 oracle,
                 directory=None,
                 project_name=None,
                 tuner_id=None,
                 overwrite=False):
        # Ops and metadata
        self.directory = directory or '.'
        self.project_name = project_name or 'untitled_project'
        if overwrite and os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)

        if not isinstance(oracle, oracle_module.Oracle):
            raise ValueError('Expected oracle to be '
                             'an instance of Oracle, got: %s' % (oracle,))
        self.oracle = oracle
        self.oracle.set_project_dir(self.directory, self.project_name, overwrite=overwrite)

        # To support tuning distribution.
        self.tuner_id = tuner_id if tuner_id is not None else 0

        # Logs etc
        self._display = Display()

        # Populate initial search space.
        hp = self.oracle.get_space()
        self.oracle.update_space(hp)

        if not overwrite and os.path.exists(self._get_tuner_fname()):
            LOGGER.info('Reloading Tuner from {}'.format(
                self._get_tuner_fname()))
            self.reload()

    def search(self, *fit_args, **fit_kwargs):
        while True:
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.run_trial(trial, *fit_args, **fit_kwargs)
            self.on_trial_end(trial)

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        raise NotImplementedError

    def on_trial_end(self, trial):
        # Send status to Logger
        self.oracle.end_trial(
            trial.trial_id, trial_module.TrialStatus.COMPLETED)
        self.oracle.update_space(trial.hyperparameters)
        self._display.on_trial_end(trial)
        self.save()

    def search_space_summary(self, extended=False):
        """Print search space summary.
        Args:
            extended: Bool, optional. Display extended summary.
                Defaults to False.
        """
        display.section('Search space summary')
        hp = self.oracle.get_space()
        display.display_setting(
            'Default search space size: %d' % len(hp.space))
        for p in hp.space:
            config = p.get_config()
            name = config.pop('name')
            display.subsection('%s (%s)' % (name, p.__class__.__name__))
            display.display_settings(config)

    def results_summary(self, num_trials=10):
        """Display tuning results summary.
        Args:
            num_trials (int, optional): Number of trials to display.
                Defaults to 10.
        """
        display.section('Results summary')
        display.display_setting('Results in %s' % self.project_dir)
        best_trials = self.oracle.get_best_trials(num_trials)
        display.display_setting('Showing %d best trials' % num_trials)
        for trial in best_trials:
            display.display_setting(
                'Objective: {} Score: {}'.format(
                    self.oracle.objective, trial.score))

    @property
    def remaining_trials(self):
        return self.oracle.remaining_trials()

    def get_state(self):
        return {}

    def set_state(self, state):
        pass

    def save(self):
        super(BaseTuner, self).save(self._get_tuner_fname())

    def reload(self):
        super(BaseTuner, self).reload(self._get_tuner_fname())

    @property
    def project_dir(self):
        dirname = os.path.join(
            self.directory,
            self.project_name)
        create_directory(dirname)
        return dirname

    def get_trial_dir(self, trial_id):
        dirname = os.path.join(
            self.project_dir,
            'trial_' + str(trial_id))
        create_directory(dirname)
        return dirname

    def _get_tuner_fname(self):
        return os.path.join(
            self.project_dir,
            'tuner_' + str(self.tuner_id) + '.json')


class PipeTuner(BaseTuner):

    def __init__(self, oracle, config, train_X, train_y, val_X, val_y,  **kwargs):
        super(PipeTuner, self).__init__(oracle, **kwargs)
        self.config = config
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        new_model_config = set_tunable_hps(self.config["ModelOption"], trial.hyperparameters)
        new_model = Recommender(new_model_config)
        scores = self.get_scores(new_model)
        self.oracle.update_trial(trial.trial_id, metrics=scores)

    def get_scores(self, model):
        _, val_loss = model.train(self.train_X, self.train_y, self.val_X, self.val_y, self.config)
        return {"mse": val_loss}
