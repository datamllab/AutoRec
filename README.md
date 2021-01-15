# AutoRec


AutoRec is a Keras-based implementation of automated recommendation algorithms for both rating prediction and Click Through Rate task. 


For more details, see the [Documentation](http://autorec.ai).


## Installation
Install from `pip`:
```
pip install autorec
```


## Quickstart
Build an rating prediction model which can search the model architecture automatically  on the MovieLens  dataset is very easy as follows:
```python
# -*- coding: utf-8 -*-
import tensorflow as tf
from autorecsys.auto_search import Search
from autorecsys.pipeline import Input, LatentFactorMapper, RatingPredictionOptimizer, ElementwiseInteraction
from autorecsys.pipeline.preprocessor import MovielensPreprocessor, NetflixPrizePreprocessor
from autorecsys.recommender import RPRecommender

# load dataset
#Movielens 1M Dataset
data = MovielensPreprocessor("./examples/datasets/ml-1m/ratings.dat")
data.preprocessing(val_test_size=0.1, random_state=1314)
train_X, train_y = data.train_X, data.train_y
val_X, val_y = data.val_X, data.val_y
test_X, test_y = data.test_X, data.test_y
user_num, item_num = data.user_num, data.item_num

# build the pipeline.
input = Input(shape=[2])
user_emb = LatentFactorMapper(column_id=0,
                              num_of_entities=user_num,
                              embedding_dim=64)(input)
item_emb = LatentFactorMapper(column_id=1,
                              num_of_entities=item_num,
                              embedding_dim=64)(input)
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
                batch_size=1024,
                epochs=10,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])
```
