# Benchmarking

Benchmarks for popular recommendation methods.

## Rating Predication Task

We adpot two dataset to evaluate our autoRec.

- **Movilens**: GroupLens Research has collected and made available rating data sets from the MovieLens web site (http://movielens.org). The data sets were collected over various periods of time, depending on the size of the set.In our experimets, we use different version of this dataset.
- **Netflix**: Netflix held the Netflix Prize open competition for the best algorithm to predict user ratings for films. The grand prize was $1,000,000 and was won by BellKor's Pragmatic Chaos team. This is the dataset that was used in that competition.

The statistics of the dataset are as follow:

|Dataset|#user|#item|#interaction|
|---|---:|---:|---:|
|[Movelens1m](#Movelens100k)|6,040|3,900|1,000,209|
|[Movelens10m](#Movelens10M)|71,567|10,681|10,000,054|
|[Movelens_latest](#Movelens_latest)|283,228|58,098|27,753,444|
|[Netflix](#netflix)|480189|480189|100480507|


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

## Movelens1M
|Movelens1M|MSE|Time(s)|
|---|---:|---:|
|MF_random|0.9589793682098389|19.953986644744873|
|MF_greedy|0.9606391191482544|21.724135398864746|
|MF_bayesian|0.9533846974372864|227.0287356376648|
|MLP_random|0.8242693543434143|308.1432795524597|
|MLP_greedy|0.8285931944847107|298.3095564842224|
|MLP_bayesian|0.8138656616210938|414.8519353866577|
|GMF_random|0.9541782736778259|23.459346294403076|
|GMF_greedy|0.9552543759346008|23.03926944732666|
|GMF_bayesian|0.9456287026405334|227.67925930023193|
|NeuMF_random|0.8197304010391235|401.256618976593|
|NeuMF_greedy|0.8222280740737915|423.58280324935913|
|NeuMF_bayesian|0.8195651173591614|722.4185743331909|
|AutoRec_random|0.8314628005027771|--|
|AutoRec_greedy|0.8159195184707642|436.2982130050659|
|AutoRec_bayesian|0.8080032467842102|--|


## Movelens10M
|Movelens10M|MSE|Time(s)|
|---|---:|---:|
|MF_random|0.8333620429039001|--|
|MF_greedy|0.8311187624931335|113.6271460056305|
|MF_bayesian|0.8149698376655579|--|
|MLP_random|0.7372392416000366|--|
|MLP_greedy|0.7254850268363953|--|
|MLP_bayesian|0.7147180438041687|2944.540461540222|
|GMF_random|0.8397544622421265|108.05639100074768|
|GMF_greedy|0.8208463191986084|107.45978927612305|
|GMF_bayesian|0.8189600706100464|1052.3578679561615|
|NeuMF_random|0.7074925303459167|2186.500470161438|
|NeuMF_greedy|0.7121766805648804|2168.569478034973|
|NeuMF_bayesian|0.707494854927063|3921.2556867599487|
|AutoRec_random|0.7167893052101135|2264.4721908569336|
|AutoRec_greedy|0.7070678472518921|1924.5541198253632|
|AutoRec_bayesian|0.7103970050811768|7049.3916828632355|



## Movelens_latest
|Movelens_latest|MSE|Time(s)|
|---|---:|---:|
|MF_random|0.7918405532836914|0.0000|
|MF_greedy|0.7755653262138367|2124.1722569465637|
|MF_bayesian|0.7719935774803162|0.0000|
|MLP_random|0.7081013917922974|18400.643934249878|
|MLP_greedy|0.7164097428321838|16887.85365796089|
|MLP_bayesian|0.7041485905647278|18703.031570911407|
|GMF_random|0.7778493165969849|1641.1448769569397|
|GMF_greedy|0.7733041644096375|1622.81472158432|
|GMF_bayesian|0.7760748267173767|0.0000|
|NeuMF_random|0.6542486548423767|0.0000|
|NeuMF_greedy|0.6556205749511719|0.0000|
|NeuMF_bayesian|0.6541292071342468|0.0000|
|AutoRec_random|0.6968541145324707|0.0000|
|AutoRec_greedy|0.7121773362159729|0.0000|
|AutoRec_bayesian|0.692548930644989|0.0000|


## Netflix
|Netflix|MSE|Time(s)|
|---|---:|---:|
|MF|0.0000|0.0000|
|MLP|0.0000|0.0000|
|GMF|0.0000|0.0000|
|NeuMF|0.0000|0.0000|
|AutoRec_random|0.0000|0.0000|
|AutoRec_bayesian|0.0000|0.0000|
|AutoRec_hyperband|0.0000|0.0000|


MSE and MAE are the mean square error and mean abslute error.

Time, for the baseline model, is the total training time; for the automated model, is the total search and training time.



## Click-Through Rate Task

We adpot two dataset to evaluate our autoRec.

- **Criteo**: Display advertising is a billion dollar effort and one of the central uses of machine learning on the Internet. However, its data and methods are usually kept under lock and key. In this research competition, CriteoLabs is sharing a week’s worth of data for you to develop models predicting ad click-through rate (CTR). Given a user and the page he is visiting, what is the probability that he will click on a given ad?
- **Avazu**: For this competition, we have provided 11 days worth of Avazu data to build and test prediction models. Can you find a strategy that beats standard classification algorithms? The winning models from this competition will be released under an open-source license.

The statistics of the dataset are as follow:

|Dataset|#user|#item|#interaction|
|---|---:|---:|---:|
|[Movielens](#Movielens)|10000|10000|10000|


|Dataset|#dense field|#sparse field|#instance|
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

## Movielens
|Movielens|Acc|NDCG|Time(s)|
|---|---:|---:|---:|
|MF|0.0000|0.0000|0.0000|
|GMF|0.0000|0.0000|0.0000|
|MLP|0.0000|0.0000|0.0000|
|NeuMF|0.0000|0.0000|0.0000|
|AutoRec_random|0.0000|0.0000|0.0000|
|AutoRec_bayesian|0.0000|0.0000|0.0000|
|AutoRec_hyperband|0.0000|0.0000|0.0000|

## Criteo
|Criteo|Acc|NDCG|Time(s)|
|---|---:|---:|---:|
|FM|0.0000|0.0000|0.0000|
|AutoRec_random|0.0000|0.0000|0.0000|
|AutoRec_bayesian|0.0000|0.0000|0.0000|
|AutoRec_hyperband|0.0000|0.0000|0.0000|


## Avazu
|Avazu|Acc|NDCG|Time(s)|
|---|---:|---:|---:|
|FM|0.0000|0.0000|0.0000|
|AutoRec_random|0.0000|0.0000|0.0000|
|AutoRec_bayesian|0.0000|0.0000|0.0000|
|AutoRec_hyperband|0.0000|0.0000|0.0000|

Accuary and NDCG are the Accuary and Normalized Discounted Cumulative Gain.

Time, for the baseline model, is the total training time; for the automated model, is the total search and training time.