# -*- coding: utf-8 -*-
# This codes are migrated from Keras Tuner: https://keras-team.github.io/keras-tuner/.  
# The copyright belows to the Keras Tuner authors.


from __future__ import absolute_import, division, print_function, unicode_literals

import random
import tensorflow as tf
import time
import json

from autorecsys.searcher.core import hyperparameters as hp_module
from autorecsys.utils import display, metric


class Stateful(object):

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def save(self, fname):
        state = self.get_state()
        state_json = json.dumps(state)
        with open(fname, 'w') as fp:
            fp.write(state_json)
        return str(fname)

    def reload(self, fname):
        with open(fname, 'r') as fp:
            state = json.load(fp)
        self.set_state(state)


class TrialStatus:
    RUNNING = 'RUNNING'
    IDLE = 'IDLE'
    INVALID = 'INVALID'
    STOPPED = 'STOPPED'
    COMPLETED = 'COMPLETED'


class Trial(Stateful):

    def __init__(self,
                 hyperparameters,
                 trial_id=None,
                 status=TrialStatus.RUNNING):
        self.hyperparameters = hyperparameters
        self.trial_id = generate_trial_id() if trial_id is None else trial_id
        self.metrics = metric.MetricsTracker()
        self.score = None
        self.best_step = None
        self.status = status

    def summary(self):
        display.section('Trial summary')
        if self.hyperparameters.values:
            display.subsection('Hp values:')
            value_need_display = {k: v for k, v in self.hyperparameters.values.items()
                                  if k in self.hyperparameters._space and
                                  self.hyperparameters._space[k].__class__.__name__ != 'Fixed'}
            display.display_settings(value_need_display)
        else:
            display.subsection('Hp values: default configuration.')
        if self.score is not None:
            display.display_setting('Score: {}'.format(self.score))
        if self.best_step is not None:
            display.display_setting('Best step: {}'.format(self.best_step))

    def get_state(self):
        return {
            'trial_id': self.trial_id,
            'hyperparameters': self.hyperparameters.get_config(),
            'metrics': self.metrics.get_config(),
            'score': self.score,
            'best_step': self.best_step,
            'status': self.status
        }

    def set_state(self, state):
        self.trial_id = state['trial_id']
        hp = hp_module.HyperParameters.from_config(
            state['hyperparameters']
        )
        self.hyperparameters = hp
        self.metrics = metric.MetricsTracker.from_config(state['metrics'])
        self.score = state['score']
        self.best_step = state['best_step']
        self.status = state['status']

    @classmethod
    def from_state(cls, state):
        trial = cls(hyperparameters=None)
        trial.set_state(state)
        return trial

    @classmethod
    def load(cls, fname):
        with tf.io.gfile.GFile(fname, 'r') as f:
            state_data = f.read()
        return cls.from_state(state_data)


def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, 1e7))
    # return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]
    return hash(s) % 1045543567
