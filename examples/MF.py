from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from autorecsys.pipeline.recommender import Recommender
from autorecsys.trainer import train
from autorecsys.utils.common import set_device
from autorecsys.pipeline.preprocesser import data_load
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # set GPU devices
    set_device('gpu:0')

    # load dataset
    train_X, train_y, val_X, val_y = data_load(dataset="movielens",
                                               dataset_path="./examples/datasets/ml-1m/ratings.dat",
                                               col_names=["user_id", "item_id", "rating", "timestamp"],
                                               used_col_names=["user_id", "item_id", "rating"],
                                               dtype={"user_id": np.int32, "item_id": np.int32, "rating": np.float32,
                                                      "timestamp": np.int32})

    # # build recommender
    config_filename = "./examples/configs/mf_config.yaml"
    recommender = Recommender(config_filename)

    # # train model
    train_loss, val_loss = recommender.train(train_X, train_y, val_X, val_y,
                                        train_config="./examples/configs/mf_config.yaml")
    # model, train_loss, val_loss  = train(model, train_X, train_y, val_X, val_y, train_config="./examples/configs/mf_config.yaml")
