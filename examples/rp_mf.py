# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import logging
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, LatentFactorMapper, RatingPredictionOptimizer, ElementwiseInteraction
from autorecsys.pipeline.preprocessor import MovielensPreprocessor, NetflixPrizePreprocessor
from autorecsys.recommender import RPRecommender

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load dataset
nf_paths = ["./examples/datasets/netflix-prize-data/combined_data_" + str(i) + ".txt" for i in range(1, 5)]
ml_1m = NetflixPrizePreprocessor(nf_paths)
# ml_1m = MovielensPreprocessor("./examples/datasets/ml-1m/ratings.dat")
# ml_1m = MovielensPreprocessor("./examples/datasets/ml-10M100K/ratings.dat")
# ml_1m = MovielensPreprocessor("./examples/datasets/ml-latest/ratings.csv", sep=',')
ml_1m.preprocessing(test_size=0.1, random_state=1314)
train_X, train_y, val_X, val_y = ml_1m.train_X, ml_1m.train_y, ml_1m.val_X, ml_1m.val_y

# build the pipeline.
input = Input(shape=[2])
user_emb = LatentFactorMapper(feat_column_id=0,
                              id_num=1000000,
                              embedding_dim=64)(input)
item_emb = LatentFactorMapper(feat_column_id=1,
                              id_num=1000000,
                              embedding_dim=28)(input)
output = ElementwiseInteraction(elementwise_type="innerporduct")([user_emb, item_emb])
output = RatingPredictionOptimizer()(output)
model = RPRecommender(inputs=input, outputs=output)

# AutoML search and predict
searcher = Search(model=model,
                  tuner='greedy',  # hyperband, greedy, bayesian
                  tuner_params={"max_trials": 5}
                  )

searcher.search(x=train_X,
                y=train_y,
                x_val=val_X,
                y_val=val_y,
                objective='val_mse',
                batch_size=10240,
                epochs = 20,
                callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)] )

logger.info('Predicted Ratings: {}'.format(searcher.predict(x=val_X)))
logger.info('Predicting Accuracy (mse): {}'.format(searcher.evaluate(x=val_X, y_true=val_y)))
