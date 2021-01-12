# -*- coding: utf-8 -*-
# This codes are migrated from Keras Tuner: https://keras-team.github.io/keras-tuner/.  
# The copyright belows to the Keras Tuner authors.


"Tuner base class."

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import copy
import inspect
import shutil
import logging
import collections

import tensorflow as tf
import numpy as np
from autorecsys.utils import display
from autorecsys.utils.common import create_directory
from autorecsys.searcher.core import trial as trial_module
from autorecsys.searcher.core import oracle as oracle_module
# from autorecsys.searcher.tuners import RandomSearch
# from autorecsys.searcher.tuners.hyperband import Hyperband

# from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
# METRIC = {'auc': roc_auc_score, 'log_loss': log_loss, 'mse': mean_squared_error}

METRIC = {'auc': tf.keras.metrics.AUC(),
          'logloss': tf.keras.metrics.CategoricalAccuracy(),
          'mse': tf.keras.metrics.MeanSquaredError(),
          'mae': tf.keras.metrics.MeanAbsoluteError(),
          'BinaryCrossentropy': tf.keras.metrics.BinaryCrossentropy(), }

# TODO: Add more extensive display.
LOGGER = logging.getLogger(__name__)


class Display(object):

    @staticmethod
    def on_trial_begin(trial):
        display.section('New model')
        trial.summary()

    @staticmethod
    def on_trial_end(trial):
        display.section('Trial complete')
        trial.summary()


class TunerCallback(tf.keras.callbacks.Callback):

    def __init__(self, tuner, trial):
        self.tuner = tuner
        self.trial = trial

    def on_epoch_begin(self, epoch, logs=None):
        self.tuner.on_epoch_begin(
            self.trial, self.model, epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.tuner.on_batch_begin(self.trial, self.model, batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.tuner.on_batch_end(self.trial, self.model, batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.tuner.on_epoch_end(
            self.trial, self.model, epoch, logs=logs)


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
                 logger=None,
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
        self.tuner_id = os.environ.get('KERASTUNER_TUNER_ID', 'tuner0')

        # Logs etc
        self.logger = logger
        self._display = Display()

        # Populate initial search space.
        hp = self.oracle.get_space()
        self.oracle.update_space(hp)

        if not overwrite and os.path.exists(self._get_tuner_fname()):
            LOGGER.info('Reloading Tuner from {}'.format(
                self._get_tuner_fname()))
            self.reload()

    def search(self, *fit_args, **fit_kwargs):
        """Performs a search for best hyperparameter configuations.
        # Arguments:
            *fit_args: Positional arguments that should be passed to
              `run_trial`, for example the training and validation data.
            *fit_kwargs: Keyword arguments that should be passed to
              `run_trial`, for example the training and validation data.
        """
        self.on_search_begin()
        while True:
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info('Oracle triggered exit')
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)
            model = self.run_trial(trial, *fit_args, **fit_kwargs)
            self.on_trial_end(trial, model)
        self.on_search_end()

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Evaluates a set of hyperparameter values.
        This method is called during `search` to evaluate a set of
        hyperparameters.
        For subclass implementers: This method is responsible for
        reporting metrics related to the `Trial` to the `Oracle`
        via `self.oracle.update_trial`.
        Simplest example:
        ```python
        def run_trial(self, trial, x, y, val_x, val_y):
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x, y)
            loss = model.evaluate(val_x, val_y)
            self.oracle.update_trial(
              trial.trial_id, {'loss': loss})
            self.save_model(trial.trial_id, model)
        ```
        # Arguments:
            trial: A `Trial` instance that contains the information
              needed to run this trial. Hyperparameters can be accessed
              via `trial.hyperparameters`.
            *fit_args: Positional arguments passed by `search`.
            *fit_kwargs: Keyword arguments passed by `search`.
        """
        raise NotImplementedError

    def save_model(self, trial_id, model, step=0):
        """Saves a Model for a given trial.
        # Arguments:
            trial_id: The ID of the `Trial` that corresponds to this Model.
            model: The trained model.
            step: For models that report intermediate results to the `Oracle`,
              the step that this saved file should correspond to. For example,
              for Keras models this is the number of epochs trained.
        """
        raise NotImplementedError

    def load_model(self, trial):
        """Loads a Model from a given trial.
        # Arguments:
            trial: A `Trial` instance. For models that report intermediate
              results to the `Oracle`, generally `load_model` should load the
              best reported `step` by relying of `trial.best_step`
        """
        raise NotImplementedError

    def on_search_begin(self):
        """A hook called at the beginning of `search`."""
        if self.logger:
            self.logger.register_tuner(self.get_state())

    def on_trial_begin(self, trial):
        """A hook called before starting each trial.
        # Arguments:
            trial: A `Trial` instance.
        """
        if self.logger:
            self.logger.register_trial(trial.trial_id, trial.get_state())

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        raise NotImplementedError

    def on_trial_end(self, trial, model):
        """A hook called after each trial is run.
        # Arguments:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        self.oracle.end_trial(
            trial.trial_id, trial_module.TrialStatus.COMPLETED)
        self.save_weights(trial, model)
        self.oracle.update_space(trial.hyperparameters)
        self._display.on_trial_end(trial)
        self.save()

    def on_search_end(self):
        """A hook called at the end of `search`."""
        if self.logger:
            self.logger.exit()

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the objective.
        This method is only a convenience shortcut. For best performance, It is
        recommended to retrain your Model on the full dataset using the best
        hyperparameters found during `search`.
        # Arguments:
            num_models (int, optional). Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.
        # Returns:
            List of trained model instances.
        """
        best_trials = self.oracle.get_best_trials(num_models)
        models = [self.load_model(trial) for trial in best_trials]
        return models

    def get_best_hyperparameters(self, num_trials=1):
        """Returns the best hyperparameters, as determined by the objective.
        This method can be used to reinstantiate the (untrained) best model
        found during the search process.
        Example:
        ```python
        best_hp = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hp)
        ```
        # Arguments:
            num_trials: (int, optional). Number of `HyperParameters` objects to
              return. `HyperParameters` will be returned in sorted order based on
              trial performance.
        # Returns:
            List of `HyperParameter` objects.
        """
        return [t.hyperparameters for t in self.oracle.get_best_trials(num_trials)]

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
            if p.__class__.__name__ == 'Fixed':
                continue
            display.subsection('%s (%s)' % (name, p.__class__.__name__))
            display.display_settings(config)

    def results_summary(self, num_trials=10):
        """Display tuning results summary.
        Args:
            num_trials (int, optional): Number of trials to display.
                Defaults to 10.
            sort_metric (str, optional): Sorting metric, when not specified
                sort models by objective value. Defaults to None.
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
        """Returns the number of trials remaining.
        Will return `None` if `max_trials` is not set.
        """
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
            str(self.tuner_id) + '.json')

    def save_weights(self, trial, model):
        raise NotImplementedError

    def load_model(self, trial):
        raise NotImplementedError


class Tuner(BaseTuner):
    """Tuner class for Keras models.
    May be subclassed to create new tuners.
    # Arguments:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        max_model_size: Int. Maximum size of weights
            (in floating point coefficients) for a valid
            models. Models larger than this are rejected.
        optimizer: Optional. Optimizer instance.
            May be used to override the `optimizer`
            argument in the `compile` step for the
            models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        loss: Optional. May be used to override the `loss`
            argument in the `compile` step for the
            models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        metrics: Optional. May be used to override the
            `metrics` argument in the `compile` step
            for the models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        directory: String. Path to the working directory (relative).
        project_name: Name to use as prefix for files saved
            by this Tuner.
        logger: Optional. Instance of Logger class, used for streaming data
            to Cloud Service for monitoring.
        overwrite: Bool, default `False`. If `False`, reloads an existing project
            of the same name if one is found. Otherwise, overwrites the project.
    """

    def __init__(self,
                 oracle,
                 max_model_size=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 directory=None,
                 project_name=None,
                 logger=None,
                 tuner_id=None,
                 overwrite=False):

        # Subclasses of `KerasHyperModel` are not automatically wrapped.
        super(Tuner, self).__init__(oracle=oracle,
                                    directory=directory,
                                    project_name=project_name,
                                    logger=logger,
                                    overwrite=overwrite)

        # Save only the last N checkpoints.
        self._save_n_checkpoints = 10

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Evaluates a set of hyperparameter values.
        This method is called during `search` to evaluate a set of
        hyperparameters.
        # Arguments:
            trial: A `Trial` instance that contains the information
              needed to run this trial. `Hyperparameters` can be accessed
              via `trial.hyperparameters`.
            *fit_args: Positional arguments passed by `search`.
            *fit_kwargs: Keyword arguments passed by `search`.
        """
        # Handle any callbacks passed to `fit`.
        fit_kwargs = copy.copy(fit_kwargs)
        callbacks = fit_kwargs.pop('callbacks', [])
        callbacks = self._deepcopy_callbacks(callbacks)
        self._configure_tensorboard_dir(callbacks, trial.trial_id)
        # `TunerCallback` calls:
        # - `Tuner.on_epoch_begin`
        # - `Tuner.on_batch_begin`
        # - `Tuner.on_batch_end`
        # - `Tuner.on_epoch_end`
        # These methods report results to the `Oracle` and save the trained Model. If
        # you are subclassing `Tuner` to write a custom training loop, you should
        # make calls to these methods within `run_trial`.
        callbacks.append(TunerCallback(self, trial))

        model = self.hypermodel.build(trial.hyperparameters)
        model.fit(*fit_args, **fit_kwargs, callbacks=callbacks)
        return model

    def save_model(self, trial_id, model, step=0):
        epoch = step
        self._checkpoint_model(model, trial_id, epoch)
        if epoch > self._save_n_checkpoints:
            self._delete_checkpoint(
                trial_id, epoch - self._save_n_checkpoints)

    def load_model(self, trial):
        model = self.hypermodel.build(trial.hyperparameters)
        # Reload best checkpoint. The Oracle scores the Trial and also
        # indicates at what epoch the best value of the objective was
        # obtained.
        best_epoch = trial.best_step
        model.load_weights(self._get_checkpoint_fname(
            trial.trial_id, best_epoch))
        return model

    def on_epoch_begin(self, trial, model, epoch, logs=None):
        """A hook called at the start of every epoch.
        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Additional metrics.
        """
        pass

    def on_batch_begin(self, trial, model, batch, logs):
        """A hook called at the start of every batch.
        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            batch: The current batch number within the
              curent epoch.
            logs: Additional metrics.
        """
        pass

    def on_batch_end(self, trial, model, batch, logs=None):
        """A hook called at the end of every batch.
        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            batch: The current batch number within the
              curent epoch.
            logs: Additional metrics.
        """
        pass

    def on_epoch_end(self, trial, model, epoch, logs=None):
        """A hook called at the end of every epoch.
        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Dict. Metrics for this epoch. This should include
              the value of the objective for this epoch.
        """
        self.save_model(trial.trial_id, model, step=epoch)
        # Report intermediate metrics to the `Oracle`.
        status = self.oracle.update_trial(
            trial.trial_id, metrics=logs, step=epoch)
        trial.status = status
        if trial.status == "STOPPED":
            model.stop_training = True

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the tuner's objective.
        The models are loaded with the weights corresponding to
        their best checkpoint (at the end of the best epoch of best trial).
        This method is only a convenience shortcut. For best performance, It is
        recommended to retrain your Model on the full dataset using the best
        hyperparameters found during `search`.
        Args:
            num_models (int, optional): Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.
        Returns:
            List of trained model instances.
        """
        # Method only exists in this class for the docstring override.
        return super(Tuner, self).get_best_models(num_models)

    def _deepcopy_callbacks(self, callbacks):
        try:
            callbacks = copy.deepcopy(callbacks)
        except:
            raise ValueError(
                'All callbacks used during a search '
                'should be deep-copyable (since they are '
                'reused across trials). '
                'It is not possible to do `copy.deepcopy(%s)`' %
                (callbacks,))
        return callbacks

    def _configure_tensorboard_dir(self, callbacks, trial_id):
        for callback in callbacks:
            # Patching tensorboard log dir
            if callback.__class__.__name__ == 'TensorBoard':
                callback.log_dir = os.path.join(
                    callback.log_dir,
                    str(trial_id))

    def _get_checkpoint_dir(self, trial_id, epoch):
        return os.path.join(
            self.get_trial_dir(trial_id),
            'checkpoints',
            'epoch_' + str(epoch))

    def _get_checkpoint_fname(self, trial_id, epoch):
        return os.path.join(
            # Each checkpoint is saved in its own directory.
            self._get_checkpoint_dir(trial_id, epoch),
            'checkpoint')

    def _checkpoint_model(self, model, trial_id, epoch):
        fname = self._get_checkpoint_fname(trial_id, epoch)
        # Save in TF format.
        model.save_weights(fname)
        return fname

    def _delete_checkpoint(self, trial_id, epoch):
        tf.io.gfile.rmtree(self._get_checkpoint_dir(trial_id, epoch))


class MultiExecutionTuner(Tuner):
    """A Tuner class that averages multiple runs of the process.
    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        executions_per_trial: Int. Number of executions
            (training a model from scratch,
            starting from a new initialization)
            to run per trial (model configuration).
            Model metrics may vary greatly depending
            on random initialization, hence it is
            often a good idea to run several executions
            per trial in order to evaluate the performance
            of a given set of hyperparameter values.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(self,
                 oracle,
                 executions_per_trial=1,
                 **kwargs):
        super(MultiExecutionTuner, self).__init__(
            oracle, **kwargs)
        if isinstance(oracle.objective, list):
            raise ValueError(
                'Multi-objective is not supported, found: {}'.format(
                    oracle.objective))
        self.executions_per_trial = executions_per_trial
        # This is the `step` that will be reported to the Oracle at the end
        # of the Trial. Since intermediate results are not used, this is set
        # to 0.
        self._reported_step = 0

    def on_epoch_end(self, trial, model, epoch, logs=None):
        # Intermediate results are not passed to the Oracle, and
        # checkpointing is handled via a `ModelCheckpoint` callback.
        pass

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_fname(
                trial.trial_id, self._reported_step),
            monitor=self.oracle.objective.name,
            mode=self.oracle.objective.direction,
            save_best_only=True,
            save_weights_only=True)

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            fit_kwargs = copy.copy(fit_kwargs)
            callbacks = fit_kwargs.pop('callbacks', [])
            callbacks = self._deepcopy_callbacks(callbacks)
            self._configure_tensorboard_dir(callbacks, trial.trial_id, execution)
            callbacks.append(TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)

            model = self.hypermodel.build(trial.hyperparameters)
            # model.summary()
            history = model.fit(*fit_args, **fit_kwargs, callbacks=callbacks)

            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == 'min':
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step)
        return model

    def _configure_tensorboard_dir(self, callbacks, trial_id, execution=0):
        for callback in callbacks:
            # Patching tensorboard log dir
            if callback.__class__.__name__ == 'TensorBoard':
                callback.log_dir = os.path.join(
                    callback.log_dir,
                    trial_id,
                    'execution{}'.format(execution))
        return callbacks


class PipeTuner(MultiExecutionTuner):

    def __init__(self, oracle, hypergraph, fit_on_val_data=False, **kwargs):
        super().__init__(oracle, **kwargs)
        self.oracle = oracle
        self.hypergraph = hypergraph
        self.need_fully_train = False
        self.best_hp = None
        self.fit_on_val_data = fit_on_val_data

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Preprocess the x and y before calling the base run_trial."""
        # Initialize new fit kwargs for the current trial.
        fit_kwargs.update(
            dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))
        new_fit_kwargs = copy.copy(fit_kwargs)

        # Preprocess the dataset and set the shapes of the HyperNodes.
        self.hypermodel = self.hypergraph.build_graphs(trial.hyperparameters)

        self._prepare_run(new_fit_kwargs)

        model = super().run_trial(trial, **new_fit_kwargs)
        return model

    def _prepare_run(self, fit_kwargs):
        validation_data = (fit_kwargs.pop('x_val', None), fit_kwargs.pop('y_val', None))

        # Update the new fit kwargs values
        fit_kwargs['x'] = fit_kwargs.get('x', None)
        fit_kwargs['y'] = fit_kwargs.get('y', None)
        fit_kwargs['validation_data'] = validation_data
        fit_kwargs['batch_size'] = fit_kwargs.get('batch_size', 32)

    def save_weights(self, trial, pipe):
        trial_dir = self.get_trial_dir(trial.trial_id)
        tf.keras.models.save_model(pipe, trial_dir)

    def load_model(self, trial):
        """Load the model in a history trial.
        # Arguments
            trial: Trial. The trial to be loaded.
        # Returns
            Tuple of (PreprocessGraph, KerasGraph, tf.keras.Model).
        """
        models = tf.keras.models.load_model(self.get_trial_dir(trial.trial_id), compile=False)
        models.compile(loss=tf.keras.losses.BinaryCrossentropy())
        self.hypermodel = None
        return models

    def get_best_model(self):
        """Load the best PreprocessGraph and Keras model.
        It is mainly used by the predict and evaluate function of AutoModel.
        # Returns
            tf.keras.Model
        """
        keras_graph = self.hypergraph.build_graphs(
            self.best_hp)
        keras_graph.reload(self.best_keras_graph_path)
        model = keras_graph.build(self.best_hp)
        model.load_weights(self.best_model_path)
        return model

    @property
    def best_keras_graph_path(self):
        return os.path.join(self.project_dir, 'best_keras_graph')

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, 'best_model')

    def _get_save_path(self, trial, name):
        filename = '{trial_id}-{name}'.format(trial_id=trial.trial_id, name=name)
        return os.path.join(self.get_trial_dir(trial.trial_id), filename)

    def on_trial_end(self, trial, model):
        """Save and clear the hypermodel."""
        super().on_trial_end(trial, model)

        self.hypermodel.save(self._get_save_path(trial, 'keras_graph'))
        self.hypermodel = None


