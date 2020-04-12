# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import logging
import numpy as np
import time
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, DenseFeatureMapper, SparseFeatureMapper, FMInteraction, MLPInteraction, PointWiseOptimizer
from autorecsys.recommender import CTRRecommender
from autorecsys.pipeline.preprocessor import CriteoPreprocessor


# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


st = time.time()
# load dataset
# criteo_path = "./examples/datasets/criteo_full/train.txt"
criteo_path = "./examples/datasets/criteo_sample_10000/train_examples.txt"
save_path = "./preprocessed_sampled_criteo_data.npy"
criteo = CriteoPreprocessor(criteo_path, save_path)
criteo.preprocessing(test_size=0.2, random_state=1314)
train_X, train_y, val_X, val_y = criteo.train_X, criteo.train_y, criteo.val_X, criteo.val_y
print("Preprocessing time:\t", time.time()-st)

# build the pipeline.
dense_input_node = Input(shape=[criteo.numer_num])
sparse_input_node = Input(shape=[criteo.categ_num])
dense_feat_emb = DenseFeatureMapper(
    num_of_fields=criteo.numer_num,
    embedding_dim=2)(dense_input_node)

# TODO: preprocess data to get sparse hash_size
sparse_feat_emb = SparseFeatureMapper(
    num_of_fields=criteo.categ_num,
    hash_size=criteo.hash_sizes,
    embedding_dim=2)(sparse_input_node)

fm_output = FMInteraction()([sparse_feat_emb])
bottom_mlp_output = MLPInteraction()([dense_feat_emb])
top_mlp_output = MLPInteraction()([fm_output, bottom_mlp_output])

output = PointWiseOptimizer()(top_mlp_output)
model = CTRRecommender(inputs=[dense_input_node, sparse_input_node], outputs=output)

# AutoML search and predict.
searcher = Search(model=model,
                  tuner='random',
                  tuner_params={'max_trials': 2, 'overwrite': True},
                  )
searcher.search(x=train_X,
                y=train_y,
                x_val=val_X,
                y_val=val_y,
                objective='val_BinaryCrossentropy',
                batch_size=10000
                )
logger.info('First 10 Predicted Ratings: {}'.format(searcher.predict(x=val_X)[:10]))
logger.info('Predicting Accuracy (logloss): {}'.format(searcher.evaluate(x=val_X, y_true=val_y)))

print("Total time:\t", time.time()-st)


# # load dataset
# mini_criteo = np.load("./examples/datasets/criteo/criteo_2M.npz")
# # TODO: preprocess train val split
# train_X = [mini_criteo['X_int'].astype(np.float32), mini_criteo['X_cat'].astype(np.float32)]
# train_y = mini_criteo['y']
# val_X, val_y = train_X, train_y
#
# # build the pipeline.
# dense_input_node = Input(shape=[13])
# sparse_input_node = Input(shape=[26])
# dense_feat_emb = DenseFeatureMapper(
#     num_of_fields=13,
#     embedding_dim=2)(dense_input_node)
#
# # TODO: preprocess data to get sparse hash_size
# sparse_feat_emb = SparseFeatureMapper(
#     num_of_fields=26,
#     hash_size=[
#         1444, 555, 175781, 128509, 306, 19,
#         11931, 630, 4, 93146, 5161, 174835,
#         3176, 28, 11255, 165206, 11, 4606,
#         2017, 4, 172322, 18, 16, 56456,
#         86, 43356
#     ],
#     embedding_dim=2)(sparse_input_node)
#
# fm_output = FMInteraction()([sparse_feat_emb])
# bottom_mlp_output = MLPInteraction()([dense_feat_emb])
# top_mlp_output = MLPInteraction()([fm_output, bottom_mlp_output])
#
# output = PointWiseOptimizer()(top_mlp_output)
# model = CTRRecommender(inputs=[dense_input_node, sparse_input_node], outputs=output)
#
# # AutoML search and predict.
# searcher = Search(model=model,
#                   tuner='random',
#                   tuner_params={'max_trials': 2, 'overwrite': True},
#                   )
# searcher.search(x=train_X,
#                 y=train_y,
#                 x_val=val_X,
#                 y_val=val_y,
#                 objective='val_BinaryCrossentropy',
#                 batch_size=10000
#                 )
# logger.info('First 10 Predicted Ratings: {}'.format(searcher.predict(x=val_X)[:10]))
# logger.info('Predicting Accuracy (logloss): {}'.format(searcher.evaluate(x=val_X, y_true=val_y)))
