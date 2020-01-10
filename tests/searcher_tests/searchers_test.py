import pytest


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('searcher_test')


def test_randomsearch(tmp_dir):
    # TODO
    pass

