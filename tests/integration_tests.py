import pytest

from autorecsys.auto_search import CFRSearch

@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test')


def test_CFRSearch(tmp_dir):
    # TODO
    pass

def test_CTRRSearch(tmp_dir):
    # TODO
    pass