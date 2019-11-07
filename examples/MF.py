# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.recommender import Recommender
from autorecsys.utils.common import set_device
from autorecsys.pipeline.preprocessor import data_load_from_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # set GPU devices
    set_device('gpu:0')

    #config filename
    config_filename = "./examples/configs/mf_config.yaml"

    # load dataset
    train_X, train_y, val_X, val_y = data_load_from_config(config_filename)

    # build recommender
    recommender = Recommender(config_filename)
    # print( recommender.summary() )
    # recommender.build(  )

    # train model
    train_loss, val_loss = recommender.train(train_X, train_y, val_X, val_y,
                                        train_config="./examples/configs/mf_config.yaml")

    # recommender.save_weights(".")
    # recommender._set_inputs(inputs)
    # recommender.save('.')