from .randomsearch import RandomSearch
from .hyperband import Hyperband

TUNER_CLASSES = {
    'random': RandomSearch,
    'hyperband': Hyperband,
}


def get_tuner_class(tuner):
    if isinstance(tuner, str) and tuner in TUNER_CLASSES:
        return TUNER_CLASSES.get(tuner)
    else:
        raise ValueError('The value {tuner} passed for argument tuner is invalid, '
                         'expected one of "random", "hyperband", '
                         '"bayesian".'.format(tuner=tuner))