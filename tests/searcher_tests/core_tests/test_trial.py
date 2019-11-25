import os
import pytest

from autorecsys.utils import metric
from autorecsys.searcher.core import hyperparameters as hps_module
from autorecsys.searcher.core import trial as trial_module

from tensorflow.keras import metrics


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('trial_test', numbered=True)


@pytest.mark.skip(reason="TODO Later")
def test_register_from_metrics():
    # As well as direction inference.
    tracker = metric.MetricsTracker(
        metrics=[metrics.CategoricalAccuracy(),
                 metrics.MeanSquaredError()]
    )
    assert set(tracker.metrics.keys()) == {'categorical_accuracy',
                                           'mean_squared_error'}
    assert tracker.metrics['categorical_accuracy'].direction == 'max'
    assert tracker.metrics['mean_squared_error'].direction == 'min'


def test_register():
    tracker = metric.MetricsTracker()
    tracker.register('new_metric', direction='max')
    assert set(tracker.metrics.keys()) == {'new_metric'}
    assert tracker.metrics['new_metric'].direction == 'max'
    with pytest.raises(ValueError,
                       match='`direction` should be one of'):
        tracker.register('another_metric', direction='wrong')
    with pytest.raises(ValueError,
                       match='already exists'):
        tracker.register('new_metric', direction='max')


def test_exists():
    tracker = metric.MetricsTracker()
    tracker.register('new_metric', direction='max')
    assert tracker.exists('new_metric')
    assert not tracker.exists('another_metric')


def test_update():
    tracker = metric.MetricsTracker()
    tracker.update('new_metric', 0.5)  # automatic registration
    assert set(tracker.metrics.keys()) == {'new_metric'}
    assert tracker.metrics['new_metric'].direction == 'min'  # default direction
    assert (tracker.get_history('new_metric') ==
            [metric.MetricObservation(0.5, step=0)])


def test_get_history():
    tracker = metric.MetricsTracker()
    tracker.update('new_metric', 0.5, step=0)
    tracker.update('new_metric', 1.5, step=1)
    tracker.update('new_metric', 2., step=2)
    assert tracker.get_history('new_metric') == [
        metric.MetricObservation(0.5, 0),
        metric.MetricObservation(1.5, 1),
        metric.MetricObservation(2., 2),
    ]
    with pytest.raises(ValueError, match='Unknown metric'):
        tracker.get_history('another_metric')


def test_get_last_value():
    tracker = metric.MetricsTracker()
    tracker.register('new_metric', 'min')
    assert tracker.get_last_value('new_metric') is None
    tracker.set_history(
        'new_metric',
        [metric.MetricObservation(1., 0),
         metric.MetricObservation(2., 1),
         metric.MetricObservation(3., 2)])
    assert tracker.get_last_value('new_metric') == 3.


def test_serialization():
    tracker = metric.MetricsTracker()
    tracker.register('metric_min', 'min')
    tracker.register('metric_max', 'max')
    tracker.set_history(
        'metric_min',
        [metric.MetricObservation(1., 0),
         metric.MetricObservation(2., 1),
         metric.MetricObservation(3., 2)])
    tracker.set_history(
        'metric_max',
        [metric.MetricObservation(1., 0),
         metric.MetricObservation(2., 1),
         metric.MetricObservation(3., 2)])

    new_tracker = metric.MetricsTracker.from_config(
        tracker.get_config())
    assert new_tracker.metrics.keys() == tracker.metrics.keys()


def test_trial():
    hps = hps_module.HyperParameters()
    hps.Int('a', 0, 10, default=3)
    trial = trial_module.Trial(
        hps, trial_id='trial1', status='COMPLETED')
    trial.metrics.register('score', direction='max')
    trial.metrics.update('score', 10, step=1)
    assert len(trial.hyperparameters.space) == 1
    _trail = trial_module.Trial.from_state(trial.get_state())
    assert _trail.hyperparameters.get('a') == 3
    assert _trail.trial_id == 'trial1'
    assert _trail.score is None
    assert _trail.best_step is None
    assert _trail.metrics.get_best_value('score') == 10
    assert _trail.metrics.get_history('score') == [metric.MetricObservation(10, step=1)]
