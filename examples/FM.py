# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import pandas as pd
import logging
import numpy as np

from autorecsys.searcher.core import hyperparameters as hp_module
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, DenseFeatureMapper, SparseFeatureMapper, MLPInteraction, FMInteraction, PointWiseOptimizer

from autorecsys.utils.common import set_device
from autorecsys.pipeline.preprocessor import Movielens1MPreprocessor, Movielens1MCTRPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fm_pipeline():
    # set GPU devices
    # set_device('cpu:0')

    # # load dataset
    mini_criteo = np.load("./tests/datasets/criteo_1000.npz")

    # TODO: preprocess train val split
    train_X = [mini_criteo['X_int'].astype(np.float32), mini_criteo['X_cat'].astype(np.float32)]
    train_y = mini_criteo['y']

    val_X, val_y = train_X, train_y

    # Build the pipeline.
    # input_node = StructuredDataInput(column_names=['user_id', 'item_id'])
    dense_input_node = Input(shape=[13])
    sparse_input_node = Input(shape=[26])
    # cpu_num should default to None.
    dense_feat_emb = DenseFeatureMapper(
                                num_of_fields=13,
                                embedding_dim=2)(dense_input_node)

    # TODO: preprocess data to get sparse hash_size
    sparse_feat_emb = SparseFeatureMapper(
                                num_of_fields=26,
                                hash_size=[
                                        1444, 555, 175781, 128509, 306, 19,
                                        11931, 630, 4, 93146, 5161, 174835,
                                        3176, 28, 11255, 165206, 11, 4606,
                                        2017, 4, 172322, 18, 16, 56456,
                                        86, 43356
                                        ],
                                embedding_dim=2)(sparse_input_node)

    fm_output = FMInteraction(embedding_dim=2)([dense_feat_emb, sparse_feat_emb])

    final_output = PointWiseOptimizer()(fm_output)

    # AutoML search and predict.
    cf_searcher = Search(
                        task='ctr',
                        tuner='random',
                        tuner_params={'max_trials': 2, 'overwrite': True},
                        inputs=[dense_input_node, sparse_input_node],
                        outputs=final_output
                        )
    cf_searcher.search(
                        x=train_X, y=train_y, x_val=val_X, y_val=val_y,
                        objective='val_BinaryCrossentropy',
                        batch_size=10000
                        )
    logger.info('First 10 Predicted Ratings: {}'.format(cf_searcher.predict(x=val_X)[:10]))
    logger.info('Predicting Accuracy (logloss): {}'.format(cf_searcher.evaluate(x=val_X, y_true=val_y)))


if __name__ == "__main__":
    fm_pipeline()
