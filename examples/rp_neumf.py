# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import logging
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, LatentFactorMapper, MLPInteraction, RatingPredictionOptimizer, \
    ElementwiseInteraction
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

#Movielens 1M Dataset
data = MovielensPreprocessor("./examples/datasets/ml-1m/ratings.dat")

##Movielens 10M Dataset
# data = MovielensPreprocessor("./examples/datasets/ml-10M100K/ratings.dat")

##Movielens latest Dataset
# data = MovielensPreprocessor("./examples/datasets/ml-latest/ratings.csv", sep=',')

data.preprocessing(val_test_size=0.1, random_state=1314)
train_X, train_y = data.train_X, data.train_y
val_X, val_y = data.val_X, data.val_y
test_X, test_y = data.test_X, data.test_y
user_num, item_num = data.user_num, data.item_num
logger.info('train_X size: {}'.format(train_X.shape))
logger.info('train_y size: {}'.format(train_y.shape))
logger.info('val_X size: {}'.format(val_X.shape))
logger.info('val_y size: {}'.format(val_y.shape))
logger.info('test_X size: {}'.format(test_X.shape))
logger.info('test_y size: {}'.format(test_y.shape))
logger.info('user total number: {}'.format(user_num))
logger.info('item total number: {}'.format(item_num))

# build the pipeline.
input = Input(shape=[2])
user_emb_gmf = LatentFactorMapper(feat_column_id=0,
                                  id_num=10000,
                                  embedding_dim=64)(input)
item_emb_gmf = LatentFactorMapper(feat_column_id=1,
                                  id_num=10000,
                                  embedding_dim=64)(input)
innerproduct_output = ElementwiseInteraction(elementwise_type="innerporduct")([user_emb_gmf, item_emb_gmf])

user_emb_mlp = LatentFactorMapper(feat_column_id=0,
                                  id_num=10000,
                                  embedding_dim=64)(input)
item_emb_mlp = LatentFactorMapper(feat_column_id=1,
                                  id_num=10000,
                                  embedding_dim=64)(input)
mlp_output = MLPInteraction()([user_emb_mlp, item_emb_mlp])

output = RatingPredictionOptimizer()([innerproduct_output, mlp_output])
model = RPRecommender(inputs=input, outputs=output)

# AutoML search and predict
searcher = Search(model=model,
                  tuner='greedy',  # random, greedy
                  tuner_params={"max_trials": 5, 'overwrite': True}
                  )
searcher.search(x=train_X,
                y=train_y,
                x_val=val_X,
                y_val=val_y,
                objective='val_mse',
                batch_size=256)
logger.info('Predicted Ratings: {}'.format(searcher.predict(x=val_X)))
logger.info('Predicting Accuracy (mse): {}'.format(searcher.evaluate(x=val_X, y_true=val_y)))
