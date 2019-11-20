import os
import pytest

from autorecsys.searcher.core.oracle import Oracle, Objective
from autorecsys.searcher.core import hyperparameters as hps_module
from autorecsys.searcher.core import trial as trial_module

from tensorflow.keras import metrics


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('oracle_test', numbered=True)


class OracleTest(Oracle):
    def _populate_space(self, trial_id):
        return {'status': trial_module.TrialStatus.IDLE,
                'values': self.hyperparameters.values}

def test_oracle(tmp_dir):
    hps = hps_module.HyperParameters()
    hps.Choice('iyo_koiyo', values=[1, 2, 3, 4, 5, 6], ordered=False)
    oracle_tst = OracleTest(objective=['mse', 'auc_roc_score'], max_trials=50, hyperparameters=hps)
    assert oracle_tst.objective == [Objective(name='mse', direction='min'), Objective(name='auc_roc_score', direction='min')]
    trial1 = oracle_tst.create_trial(tuner_id='114514')
    trial2 = oracle_tst.create_trial(tuner_id='114514')
    oracle_tst.set_project_dir(directory=tmp_dir, project_name='test', overwrite=False)
    oracle_tst.save()
    assert os.path.exists(os.path.join(tmp_dir, oracle_tst._get_oracle_fname()))
    oracle_tst._save_trial(trial1)
    oracle_tst._save_trial(trial2)
    assert os.path.exists(os.path.join(oracle_tst._project_dir, f'trial_{trial1.trial_id}'))
    assert os.path.exists(os.path.join(oracle_tst._project_dir, f'trial_{trial2.trial_id}'))
    oracle_tst.reload()
    assert all(_id in oracle_tst.trials for _id in [trial1.trial_id, trial2.trial_id])