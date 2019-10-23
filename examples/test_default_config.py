from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.recommender import Recommender
from autorecsys.utils import load_config, extract_tunable_hps

if __name__ == "__main__":

    # build recommender
    config_filename = "block_default_configs"
    config = load_config(config_filename)
    print(config)

