from .randomsearch import RandomSearch
from .bayesian import BayesianOptimization
from .greedy import Greedy

TUNER_CLASSES = {
    'random': RandomSearch,
    'bayesian': BayesianOptimization,
    "greedy": Greedy
}


def get_tuner_class(tuner):
    if isinstance(tuner, str) and tuner in TUNER_CLASSES:
        return TUNER_CLASSES.get(tuner)
    else:
        raise ValueError('The value {tuner} passed for argument tuner is invalid, '
                         'expected one of "random","bayesian".'.format(tuner=tuner))
