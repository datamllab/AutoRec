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

- **MF**:Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.
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
|Movelens1M|Val_MSE|Test_MSE|Time(s)|
|---|---:|---:|---:|
|MF_random|0.7553643584251404|0.7550543546676636|103.57773876190186|
|MF_greedy|0.7503780722618103|0.7502530217170715|85.47167634963989|
|MF_bayesian|0.7521297335624695|0.7517699599266052|1031.2954790592194|
|MLP_random|0.7676995396614075|0.7681054472923279|1383.5030148029327|
|MLP_greedy|0.769902765750885|0.7706407308578491|1292.7048692703247|
|MLP_bayesian|0.758850634098053|0.7597852945327759|1353.2627713680267|
|NeuMF_random|0.7707042694091797|0.7720282077789307|1025.5578093528748|
|NeuMF_greedy|0.7517987489700317|0.7520723342895508|1276.7933542728424|
|NeuMF_bayesian|0.7721487879753113|0.7723560333251953|1098.1503052711487|
|AutoRec_random|0.7500635981559753|0.749731719493866|1577.6531774997711|
|AutoRec_greedy|0.7496007084846497|0.7510735392570496|1689.560632944107|
|AutoRec_bayesian|0.7484513521194458|0.7494882345199585|5405.682264328003|


## Movelens10M
|Movelens10M|Val_MSE|Test_MSE|Time(s)|
|---|---:|---:|---:|
|MF_random|0.6472423672676086|0.6456527709960938|795.4746537208557|
|MF_greedy|0.6473642587661743|0.6467021107673645|838.2489671707153|
|MF_bayesian|0.6490539312362671|0.6481097936630249|7755.805980920792|
|MLP_random||||
|MLP_greedy|0.6532657742500305|0.652294397354126|10709.204501867294|
|MLP_bayesian||||
|NeuMF_random|0.6536459922790527|0.6527888774871826|16713.71854186058|
|NeuMF_greedy|0.6541951298713684|0.6537747979164124|11205.822769880295|
|NeuMF_bayesian|0.650793194770813|0.6504989862442017|15727.56122994423|
|AutoRec_random||||
|AutoRec_greedy||||
|AutoRec_bayesian||||




## Movelens_latest
|Movelens_latest|Val_MSE|Test_MSE|Time(s)|
|---|---:|---:|---:|
|MF_random|0.6520289182662964|0.6528090238571167|68519.18232417107|
|MF_greedy||||
|MF_bayesian||||
|MLP_random||||
|MLP_greedy||||
|MLP_bayesian||||
|NeuMF_random||||
|NeuMF_greedy|0.6434351801872253|0.6440964937210083|56383.871745824814|
|NeuMF_bayesian||||
|AutoRec_random|0.6365838050842285|0.6371557712554932|133145.96114301682|
|AutoRec_greedy||||
|AutoRec_bayesian|0.6448036432266235|0.6453331708908081|133532.19134521484|


## Netflix
|Netflix|Val_MSE|Test_MSE|Time(s)|
|---|---:|---:|---:|
|MF_random|0.7473645806312561|0.74784255027771|8169.921831846237|
|MF_greedy|0.7397633790969849|0.7402286529541016|8646.685072422028|
|MF_bayesian|0.7282611727714539|0.7287141680717468|82759.47434949875|
|MLP_random|0.7549719214439392| 0.7553735971450806|59066.82922792435|
|MLP_greedy|0.7648082375526428|0.7652896046638489|56700.0296475887s3|
|MLP_bayesian|0.7546935081481934|0.755224347114563|46708.42347598076|
|NeuMF_random|0.7073774337768555|0.7063089609146118|50333.9074454409|
|NeuMF_greedy|0.6434351801872253|0.6440964937210083|56383.871745824814|
|NeuMF_bayesian|0.70604610443s11523|0.706568717956543|73228.66933822632|
|AutoRec_random|0.6365838050842285|0.6371557712554932|133145.96114301682|
|AutoRec_greedy|0.739780068397522|0.7401751279830933|105307.948792696|
|AutoRec_bayesian|0.6448036432266235|0.6453331708908081|133532.19134521484|


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
|Movielens|logloss|AUC|Time(s)|
|---|---:|---:|---:|
|MF|0.0000|0.0000|0.0000|
|GMF|0.0000|0.0000|0.0000|
|MLP|0.0000|0.0000|0.0000|
|NeuMF|0.0000|0.0000|0.0000|
|AutoRec_random|0.0000|0.0000|0.0000|
|AutoRec_bayesian|0.0000|0.0000|0.0000|
|AutoRec_hyperband|0.0000|0.0000|0.0000|

## Criteo
|Criteo|logloss|AUC|Time(s)|
|---|---:|---:|---:|
|FM|0.0000|0.0000|0.0000|
|AutoRec_random|0.0000|0.0000|0.0000|
|AutoRec_bayesian|0.0000|0.0000|0.0000|
|AutoRec_hyperband|0.0000|0.0000|0.0000|


## Avazu
|Avazu|logloss|AUC|Time(s)|
|---|---:|---:|---:|
|FM|0.0000|0.0000|0.0000|
|AutoRec_random|0.0000|0.0000|0.0000|
|AutoRec_bayesian|0.0000|0.0000|0.0000|
|AutoRec_hyperband|0.0000|0.0000|0.0000|

Logloss and AUC are the binary cross-entropy loss and Area Under the Receiver Operating Characteristic Curve Score.

Time, for the baseline model, is the total training time; for the automated model, is the total search and training time.