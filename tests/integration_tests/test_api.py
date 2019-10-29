import pytest

from autorecsys.api import Job, AutoSearch
from autorecsys.pipeline.recommender import Recommender


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('api_test')


def test_job_api(tmp_dir):
    config_filename = "./tests/configs/job_config.yaml"
    tmp_job = Job(config_filename)
    tmp_job.run()
    assert type(tmp_job.model) == Recommender


def test_search_api(tmp_dir):
    config_filename = "./tests/configs/random_search_config.yaml"
    search_api = AutoSearch(config_filename)
    final_results = search_api.search()
    # TODO
    pass
