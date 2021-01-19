# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import logging
import tensorflow as tf
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, LatentFactorMapper, MLPInteraction, PointWiseOptimizer
from autorecsys.pipeline.interactor import InnerProductInteraction
from autorecsys.recommender import CTRRecommender
from autorecsys.pipeline.preprocessor import CriteoPreprocessor


# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load dataset
criteo = CriteoPreprocessor()  # automatically set up for preprocessing the Criteo dataset
train_X, train_y, val_X, val_y, test_X, test_y = criteo.preprocess()

# build the pipeline.
input = Input(shape=[criteo.get_categorical_count()])
user_emb_gmf = LatentFactorMapper(feat_column_id=0,
                                  id_num=10000,
                                  embedding_dim=64)(input)
item_emb_gmf = LatentFactorMapper(feat_column_id=1,
                                  id_num=10000,
                                  embedding_dim=64)(input)

user_emb_mlp = LatentFactorMapper(feat_column_id=0,
                                  id_num=10000,
                                  embedding_dim=64)(input)
item_emb_mlp = LatentFactorMapper(feat_column_id=1,
                                  id_num=10000,
                                  embedding_dim=64)(input)
innerproduct_output = InnerProductInteraction()([user_emb_gmf, item_emb_gmf])
mlp_output = MLPInteraction()([user_emb_mlp, item_emb_mlp])
output = PointWiseOptimizer()([innerproduct_output, mlp_output])
model = CTRRecommender(inputs=input, outputs=output)

# AutoML search and predict.
searcher = Search(model=model,
                  tuner='random',
                  tuner_params={'max_trials': 10, 'overwrite': True},
                  )
searcher.search(x=[criteo.get_x_categorical(train_X)],
                y=train_y,
                x_val=[criteo.get_x_categorical(val_X)],
                y_val=val_y,
                objective='val_BinaryCrossentropy',
                batch_size=256,
                epochs = 2,
                callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)] 
                )
logger.info('Predicted Ratings: {}'.format(searcher.predict(x=[criteo.get_x_categorical(val_X)])))
logger.info('Predicting Accuracy (mse): {}'.format(searcher.evaluate(x=[criteo.get_x_categorical(val_X)], y_true=val_y)))
