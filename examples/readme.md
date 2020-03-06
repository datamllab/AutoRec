# Benchmarking

Benchmarks for popular recommendation methods.

## Rating Predication Task

We adpot two dataset to evaluate our autoRec.

- **Movilens**: GroupLens Research has collected and made available rating data sets from the MovieLens web site (http://movielens.org). The data sets were collected over various periods of time, depending on the size of the set.In our experimets, we use different version of this dataset.
- **Netflix**: Netflix held the Netflix Prize open competition for the best algorithm to predict user ratings for films. The grand prize was $1,000,000 and was won by BellKor's Pragmatic Chaos team. This is the dataset that was used in that competition.

The statistic of the dataset are as follow:

|Dataset|#user|#item|#interaction|
|---|---:|---:|---:|
|[Movelens100k](#Movelens100k)|10000|10000|10000|
|[Movelens1m](#ml1m)|10000|10000|10000|
|[Movelens_latest](#mllatest)|10000|10000|10000|
|[Netflix](#netflix)|10000|10000|10000|


Some popular model for rating prediction:

- **MF**:Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.[
- **MLP**: Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling.
- **GMF**: Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling.
- **NeuMF**: Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling.
- **Hp search**:  Our autorec with hyperparameter search. 
- **Block Search**: Our autorec with both the block search and hyperparameter search. 

All benchmarks were run with our AutoRec Package. 
The benchmarks experiments were run on a machine with dual
Intel Xeon E5-2630 v3 processors (8 cores each plus hyperthreading means 32
threads) and one GTX 2080Ti running Ubuntu 16.04 with the Tensorflow 2.1.0 and CUDA 10.0 Release.

We benchmark all models with a minibatch size of 256 ;
this allows fair comparisons between different models.

The following models are benchmarked:

## Movelens100k
|Movelens100k|MSE|MAE|Time(s)|
|---|---:|---:|---:|
|MF|0.0000|0.0000|0.0000|
|MLP|0.0000|0.0000|0.0000|
|GMF|0.0000|0.0000|0.0000|
|NeuMF|0.0000|0.0000|0.0000|
|Hp search|0.0000|0.0000|0.0000|
|Block Search|0.0000|0.0000|0.0000|


## Movelens1M
|Movelens1M|MSE|MAE|Time(s)|
|---|---:|---:|---:|
|MF|0.0000|0.0000|0.0000|
|MLP|0.0000|0.0000|0.0000|
|GMF|0.0000|0.0000|0.0000|
|NeuMF|0.0000|0.0000|0.0000|
|Hp search|0.0000|0.0000|0.0000|
|Block Search|0.0000|0.0000|0.0000|


## Movelens_latest
|Movelens_latest|MSE|MAE|Time(s)|
|---|---:|---:|---:|
|MF|0.0000|0.0000|0.0000|
|MLP|0.0000|0.0000|0.0000|
|GMF|0.0000|0.0000|0.0000|
|NeuMF|0.0000|0.0000|0.0000|
|Hp search|0.0000|0.0000|0.0000|
|Block Search|0.0000|0.0000|0.0000|


## Netflix
|Netflix|MSE|MAE|Time(s)|
|---|---:|---:|---:|
|MF|0.0000|0.0000|0.0000|
|MLP|0.0000|0.0000|0.0000|
|GMF|0.0000|0.0000|0.0000|
|NeuMF|0.0000|0.0000|0.0000|
|Hp search|0.0000|0.0000|0.0000|
|Block Search|0.0000|0.0000|0.0000|


MSE and MAE are the mean square error and mean abslute error.

Time, for the baseline model, is the total training time; for the automated model, is the total search and training time.



## Click-Through Rate Task

We adpot two dataset to evaluate our autoRec.

- **Criteo**: Display advertising is a billion dollar effort and one of the central uses of machine learning on the Internet. However, its data and methods are usually kept under lock and key. In this research competition, CriteoLabs is sharing a week’s worth of data for you to develop models predicting ad click-through rate (CTR). Given a user and the page he is visiting, what is the probability that he will click on a given ad?
- **Avazu**: For this competition, we have provided 11 days worth of Avazu data to build and test prediction models. Can you find a strategy that beats standard classification algorithms? The winning models from this competition will be released under an open-source license.

The statistic of the dataset are as follow:

|Dataset|#user|#item|#interaction|
|---|---:|---:|---:|
|[Criteo](#Criteo)|10000|10000|10000|
|[Avazu](#Avazu)|10000|10000|10000|


Some popular model for rating prediction:

- **NeuMF**: Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling.
- **Hp search**:  Our autorec with hyperparameter search. 
- **Block Search**: Our autorec with both the block search and hyperparameter search. 

All benchmarks were run with our AutoRec Package. 
The benchmarks experiments were run on a machine with dual
Intel Xeon E5-2630 v3 processors (8 cores each plus hyperthreading means 32
threads) and one GTX 2080Ti running Ubuntu 16.04 with the Tensorflow 2.1.0 and CUDA 10.0 Release.

We benchmark all models with a minibatch size of 256; this allows fair comparisons between different models.

The following models are benchmarked:

## Criteo
|Movelens100k|Acc|NDCG|Time(s)|
|---|---:|---:|---:|
|NeuMF|0.0000|0.0000|0.0000|
|Hp search|0.0000|0.0000|0.0000|
|Block Search|0.0000|0.0000|0.0000|


## Avazu
|Movelens1M|Acc|NDCG|Time(s)|
|---|---:|---:|---:|
|NeuMF|0.0000|0.0000|0.0000|
|Hp search|0.0000|0.0000|0.0000|
|Block Search|0.0000|0.0000|0.0000|


Accuary and NDCG are the Accuary and Normalized Discounted Cumulative Gain.

Time, for the baseline model, is the total training time; for the automated model, is the total search and training time.