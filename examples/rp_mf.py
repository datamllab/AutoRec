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
from autorecsys.pipeline import Input, LatentFactorMapper, InnerProductInteraction, RatingPredictionOptimizer
from autorecsys.pipeline.preprocessor import MovielensPreprocessor
from autorecsys.recommender import RPRecommender

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load dataset
##Netflix Dataset
# dataset_paths = ["./examples/datasets/netflix-prize-data/combined_data_" + str(i) + ".txt" for i in range(1, 5)]
# data = NetflixPrizePreprocessor(dataset_paths)

# Step 1: Preprocess data
movielens = MovielensPreprocessor()
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()
train_X_categorical = movielens.get_x_categorical(train_X)
val_X_categorical = movielens.get_x_categorical(val_X)
test_X_categorical = movielens.get_x_categorical(test_X)
user_num, item_num = movielens.get_hash_size()

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
input = Input(shape=[2])
user_emb = LatentFactorMapper(column_id=0,
                              num_of_entities=user_num,
                              embedding_dim=64)(input)
item_emb = LatentFactorMapper(column_id=1,
                              num_of_entities=item_num,
                              embedding_dim=64)(input)

# Step 2.2: Setup interactors to handle models
output = InnerProductInteraction()([user_emb, item_emb])

# Step 2.3: Setup optimizer to handle the target task
output = RatingPredictionOptimizer()(output)
model = RPRecommender(inputs=input, outputs=output)

# Step 3: Build the searcher, which provides search algorithm
searcher = Search(model=model,
                  tuner='greedy',  # hyperband, greedy, bayesian
                  tuner_params={"max_trials": 5}
                  )

# Step 4: Use the searcher to search the recommender
searcher.search(x=[train_X_categorical],
                y=train_y,
                x_val=[val_X_categorical],
                y_val=val_y,
                objective='val_mse',
                batch_size=1024,
                epochs=10,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])
logger.info('Validation Accuracy (mse): {}'.format(searcher.evaluate(x=val_X_categorical,
                                                                     y_true=val_y)))

# Step 5: Evaluate the searched model
logger.info('Test Accuracy (mse): {}'.format(searcher.evaluate(x=test_X_categorical,
                                                               y_true=test_y)))
