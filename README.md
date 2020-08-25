# MD-ASES
Codes for dataset manipulation, hyperparameters training and tests of MD-ASES: Multi-document - Aspect-based Sentiment Extractive Summarizer. 

MD-ASES is a sentiment aware extractive summarization framework for multiple user reviews on topics such as products or businesses, generally found on Sentiment Analysis datasets. This work was developed in [Centro de Ciência de Dados (C2D) lab](http://c2d.poli.usp.br/), a partnership between POLI - USP and Itaú-Unibanco. The framework will be published on [ENIAC 2020 conference](http://www2.sbc.org.br/bracis2020/eniac.html) under the title of "A Framework for Multi-document Extractive Summarization of Reviews with Aspect-based Sentiment Analysis".

This repository contains python scripts for the experimental setup described in the paper:

* Preparation of Yelp Academic dataset on *data_prep.py*;

* TF-IDF aspect identification module on *aspect.py*;

* Naive-Bayes SA model implementation on *sentiment.py*;

* Naive-Bayes SA model parameters tunning on *train.py*;

* Experiments as described in the Experimental setup on *experiments.py*. 

## Download Yelp Academic dataset

## data_prep.py

## aspect.py

## sentiment.py

## train.py

## experiments.py

Obs.: These scripts were develloped with the exclusive porpuse of generating experimental results for the above cited paper. Therefore, our implementations of classic AI methods (such as Naive-Bayes classification or TF-IDF aspect identification) or data preparation weren't designed to be generic, reusable, optimized, etc. Feel free to use it though :)  