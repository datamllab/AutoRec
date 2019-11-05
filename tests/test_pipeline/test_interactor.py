from __future__ import absolute_import, division, print_function, unicode_literals

import math
import pytest
import tensorflow as tf

import autorecsys.pipeline.interactor as interactor

@pytest.fixture
def ractname():
    return {"a":"1"}


def test_set(ractname):
    print("")
    #assert interactor.set_interactor_from_config("MLP", ractname) == 

if __name__ == "__main__":
    #test_set(ractname)
    print(interactor.set_interactor_from_config("MLP", ractname))