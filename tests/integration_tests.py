import pytest

from autorecsys.auto_search import Search

@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test')


def test_Search(tmp_dir):
    # TODO
    pass