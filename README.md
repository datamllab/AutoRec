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
# load dataset
ml_1m = MovielensPreprocessor("./examples/datasets/ml-1m/ratings.dat")
ml_1m.preprocessing(test_size=0.1, random_state=1314)
train_X, train_y, val_X, val_y = ml_1m.train_X, ml_1m.train_y, ml_1m.val_X, ml_1m.val_y

# build the pipeline.
input = Input(shape=[2])
user_emb = LatentFactorMapper(feat_column_id=0,
                              id_num=10000,
                              embedding_dim=64)(input)
item_emb = LatentFactorMapper(feat_column_id=1,
                              id_num=10000,
                              embedding_dim=64)(input)
output = ElementwiseInteraction(elementwise_type="innerporduct")([user_emb, item_emb])
output = RatingPredictionOptimizer()(output)
model = RPRecommender(inputs=input, outputs=output)

# AutoML search and predict
cf_searcher = Search(model=model,
                     tuner='greedy',  # hyperband, greedy, bayesian
                     tuner_params={"max_trials": 5}
                     )
cf_searcher.search(x=train_X,
                   y=train_y,
                   x_val=val_X,
                   y_val=val_y,
                   objective='val_mse',
                   batch_size=1024)
```
