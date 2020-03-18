from .randomsearch import RandomSearch
<<<<<<< HEAD
from .hyperband import Hyperband

TUNER_CLASSES = {
    'random': RandomSearch,
    'hyperband': Hyperband,
=======
from .bayesian import BayesianOptimization
from .greedy import Greedy

TUNER_CLASSES = {
    'random': RandomSearch,
    'bayesian': BayesianOptimization,
    "greedy": Greedy
>>>>>>> a2016bff64c0aac2bec78c28c2273be59b94717c
}


def get_tuner_class(tuner):
    if isinstance(tuner, str) and tuner in TUNER_CLASSES:
        return TUNER_CLASSES.get(tuner)
    else:
        raise ValueError('The value {tuner} passed for argument tuner is invalid, '
<<<<<<< HEAD
                         'expected one of "random", "hyperband", '
                         '"bayesian".'.format(tuner=tuner))
=======
                         'expected one of "random","bayesian".'.format(tuner=tuner))
>>>>>>> a2016bff64c0aac2bec78c28c2273be59b94717c
