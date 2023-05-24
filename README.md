# Overview

This is the code for the paper _Sample Attackability in Natural Language Adversarial Attacks_. This paper has been accepted at the [ACL 2023 Workshop, TrustNLP: Third Workshop on Trustworthy Natural Language Processing](https://trustnlpworkshop.github.io/).

## Abstract

Adversarial attack research in natural language processing (NLP) has made significant progress in designing powerful attack methods and defence approaches. However, few efforts have sought to identify which source samples are the most attackable or robust, i.e. can we determine for an unseen target model, which samples are the most vulnerable to an adversarial attack. This work formally extends the definition of sample attackability/robustness for NLP attacks. Experiments on two popular NLP datasets, four state of the art models and four different NLP adversarial attack methods, demonstrate that sample uncertainty is insufficient for describing characteristics of attackable/robust samples and hence a deep learning based detector can perform much better at identifying the most attackable and robust samples for an unseen target model. Nevertheless, further analysis finds that there is little agreement in which samples are considered the most attackable/robust across different NLP attack methods, explaining a lack of portability of attackability detection methods across attack methods.


# Using the code

## Requirements

`pip install transformers textattack datasets tensorflow_hub tensorflow`

Clone this repository.

## Running scripts

The following pipeline demonstrates how to use the scripts to reproduce the results in the paper. For each step and example command is given - the passed arguments can be changed as desired.

1) Train: `python train.py --out_dir experiments/trained_models --model_name xlnet-base-cased --data_name twitter --bs 8 --epochs 3 --lr 1e-5 --seed 1 --num_classes 6`
