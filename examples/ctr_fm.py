# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import logging
import numpy as np
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, DenseFeatureMapper, SparseFeatureMapper, FMInteraction, PointWiseOptimizer
from autorecsys.recommender import CTRRecommender

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load dataset
mini_criteo = np.load("./examples/datasets/criteo_1000.npz")
# TODO: preprocess train val split
train_X = [mini_criteo['X_int'].astype(np.float32), mini_criteo['X_cat'].astype(np.float32)]
train_y = mini_criteo['y']
val_X, val_y = train_X, train_y

# build the pipeline.
dense_input_node = Input(shape=[13])
sparse_input_node = Input(shape=[26])
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
output = FMInteraction(embedding_dim=2)([dense_feat_emb, sparse_feat_emb])
output = PointWiseOptimizer()(output)
model = CTRRecommender(inputs=[dense_input_node, sparse_input_node], outputs=output)

# AutoML search and predict.
cf_searcher = Search(model=model,
                     tuner='random',
                     tuner_params={'max_trials': 2, 'overwrite': True},
                     )
cf_searcher.search(x=train_X,
                   y=train_y,
                   x_val=val_X,
                   y_val=val_y,
                   objective='val_BinaryCrossentropy',
                   batch_size=10000
                   )
logger.info('First 10 Predicted Ratings: {}'.format(cf_searcher.predict(x=val_X)[:10]))
logger.info('Predicting Accuracy (logloss): {}'.format(cf_searcher.evaluate(x=val_X, y_true=val_y)))
