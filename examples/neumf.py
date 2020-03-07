# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import logging
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, LatentFactorMapper, MLPInteraction, PointWiseOptimizer, ElementwiseInteraction
from autorecsys.pipeline.preprocessor import Movielens1MCTRPreprocessor

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load dataset
ml_1m = Movielens1MCTRPreprocessor("./examples/datasets/ml-1m/ratings.dat")
ml_1m.preprocessing(test_size=0.1, random_state=1314, num_neg=10, mult=2)
train_X, train_y, val_X, val_y = ml_1m.train_X, ml_1m.train_y, ml_1m.val_X, ml_1m.val_y

# Build the pipeline.
input = Input(shape=[2])
user_emb_gmf = LatentFactorMapper(feat_column_id=0,
                                  id_num=10000,
                                  embedding_dim=10)(input)
item_emb_gmf = LatentFactorMapper(feat_column_id=1,
                                  id_num=10000,
                                  embedding_dim=10)(input)

user_emb_mlp = LatentFactorMapper(feat_column_id=0,
                                  id_num=10000,
                                  embedding_dim=10)(input)
item_emb_mlp = LatentFactorMapper(feat_column_id=1,
                                  id_num=10000,
                                  embedding_dim=10)(input)
innerproduct_output = ElementwiseInteraction(elementwise_type="innerporduct")([user_emb_gmf, item_emb_gmf])
mlp_output = MLPInteraction()([user_emb_mlp, item_emb_mlp])
output = PointWiseOptimizer()([innerproduct_output, mlp_output])

# AutoML search and predict.
cf_searcher = Search(tuner='random',
                     tuner_params={'max_trials': 10, 'overwrite': True},
                     inputs=input,
                     outputs=output)
cf_searcher.search(x=train_X,
                   y=train_y,
                   x_val=val_X,
                   y_val=val_y,
                   objective='val_BinaryCrossentropy',
                   batch_size=256)
logger.info('Predicted Ratings: {}'.format(cf_searcher.predict(x=val_X)))
logger.info('Predicting Accuracy (mse): {}'.format(cf_searcher.evaluate(x=val_X, y_true=val_y)))
