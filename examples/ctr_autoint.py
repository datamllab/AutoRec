# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import logging
import tensorflow as tf
import numpy as np
# from autorecsys.auto_search import Search
from autorecsys.pipeline import DenseFeatureMapper, SparseFeatureMapper, SelfAttentionInteraction, MLPInteraction, \
    PointWiseOptimizer
# from autorecsys.pipeline import Input
# from autorecsys.recommender import CTRRecommender
import numpy as np
import autokeras as ak
import tensorflow as tf

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# sync
# load dataset
mini_criteo = np.load("datasets/criteo_ctr/criteo_1000.npz")
# TODO: preprocess train val split
train_X = [mini_criteo['X_int'].astype(np.float32), mini_criteo['X_cat'].astype(np.float32)]
train_y = mini_criteo['y']
val_X, val_y = train_X, train_y
# build the pipeline.
dense_input_node = ak.Input(shape=[13])
sparse_input_node = ak.Input(shape=[26])
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

attention_output = SelfAttentionInteraction()([dense_feat_emb, sparse_feat_emb])
bottom_mlp_output = MLPInteraction()([dense_feat_emb])
top_mlp_output = MLPInteraction()([attention_output, bottom_mlp_output])

# output = PointWiseOptimizer()(top_mlp_output)
output = ak.ClassificationHead()(top_mlp_output)
# model = CTRRecommender(inputs=[dense_input_node, sparse_input_node], outputs=output)

auto_model = ak.AutoModel(inputs=[dense_input_node, sparse_input_node],
                          outputs=output,
                          overwrite=True,
                          max_trials=2)
auto_model.fit(train_X, train_y, epochs=1)
print(auto_model.evaluate(val_X, val_y))

# AutoML search and predict.
# searcher = Search(model=model,
#                   tuner='random',
#                   tuner_params={'max_trials': 2, 'overwrite': True},
#                   )
# searcher.search(x=train_X,
#                 y=train_y,
#                 x_val=val_X,
#                 y_val=val_y,
#                 objective='val_BinaryCrossentropy',
#                 batch_size=10000,
#                 epochs = 20,
#                 callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)]
#                 )
# logger.info('First 10 Predicted Ratings: {}'.format(searcher.predict(x=val_X)[:10]))
# logger.info('Predicting Accuracy (logloss): {}'.format(searcher.evaluate(x=val_X, y_true=val_y)))
