# Overview

This is the code for the paper _Sample Attackability in Natural Language Adversarial Attacks_. This paper has been accepted at the [ACL 2023 Workshop, TrustNLP: Third Workshop on Trustworthy Natural Language Processing](https://trustnlpworkshop.github.io/).

## Abstract

Adversarial attack research in natural language processing (NLP) has made significant progress in designing powerful attack methods and defence approaches. However, few efforts have sought to identify which source samples are the most attackable or robust, i.e. can we determine for an unseen target model, which samples are the most vulnerable to an adversarial attack. This work formally extends the definition of sample attackability/robustness for NLP attacks. Experiments on two popular NLP datasets, four state of the art models and four different NLP adversarial attack methods, demonstrate that sample uncertainty is insufficient for describing characteristics of attackable/robust samples and hence a deep learning based detector can perform much better at identifying the most attackable and robust samples for an unseen target model. Nevertheless, further analysis finds that there is little agreement in which samples are considered the most attackable/robust across different NLP attack methods, explaining a lack of portability of attackability detection methods across attack methods.


# Using the code

## Requirements

`pip install transformers textattack datasets tensorflow_hub tensorflow`

Clone this repository.

## Running scripts

The following pipeline demonstrates how to use the scripts to reproduce the results in the paper. For each step an example command is given - the passed arguments can be changed as needed.

1. **Train standard classifier**: `python train.py --out_dir experiments/trained_models --model_name xlnet-base-cased --data_name twitter --bs 8 --epochs 3 --lr 1e-5 --seed 1 --num_classes 6`
2. **Evaluate classifier performance**: `python eval.py --model_path_base experiments/trained_models/google-electra-base-discriminator_twitter_pretrainedTrue_seed --model_name google/electra-base-discriminator --data_name twitter --bs 8 --num_seeds 1 --num_classes 6`
3. **Save classifer predictions for val/test set**: `python predict.py --out_dir experiments/predictions/test --model_path experiments/trained_models/google-electra-base-discriminator_twitter_pretrainedTrue_seed1.th --model_name google/electra-base-discriminator --data_name twitter --num_classes 6`
4. **Attack the classifier and save the smallest perturbation sizes required for successful attack**: `python attack.py --out_dir experiments/perturbations/val/batches --model_path experiments/trained_models/roberta-base_twitter_pretrainedTrue_seed1.th --model_name roberta-base --data_name twitter --num_classes 6 --attack_method textfooler --batch --val --min_pert --start 960 --end 1000`. Note that a separate script can be used to combine the bathes of perturbation sizes into a single file
5. **Train sample attackability/robustness detector**: `python trn_attackability_det.py --out_dir experiments/trained_models/robust --model_name fcn-roberta-base --data_name twitter --perts experiments/perturbations/val/textfooler_roberta-base_twitter.pt --trained_model_path experiments/trained_models/roberta-base_twitter_pretrainedTrue_seed1.th --thresh 0.35 --bs 8 --epochs 5 --sch 3 --lr 1e-5 --seed 1 --num_classes 6 --robust`
6. **Evaluate detector performance**: `python evl_attackability_det.py --model_paths experiments/trained_models/robust/robust_thresh0.35_fcn-bert-base-uncased_twitter_seed1.th experiments/trained_models/robust/robust_thresh0.35_fcn-roberta-base_twitter_seed1.th experiments/trained_models/robust/robust_thresh0.35_fcn-xlnet-base-cased_twitter_seed1.th --model_names fcn-bert-base-uncased fcn-roberta-base fcn-xlnet-base-cased --data_name twitter --perts experiments/perturbations/test/textfooler_bert-base-uncased_twitter.pt experiments/perturbations/test/textfooler_roberta-base_twitter.pt experiments/perturbations/test/textfooler_xlnet-base-cased_twitter.pt experiments/perturbations/test/textfooler_google-electra-base-discriminator_twitter.pt --thresh 0.35 --bs 8 --trained_model_paths experiments/trained_models/bert-base-uncased_twitter_pretrainedTrue_seed1.th experiments/trained_models/roberta-base_twitter_pretrainedTrue_seed1.th experiments/trained_models/xlnet-base-cased_twitter_pretrainedTrue_seed1.th --num_classes 6`. This command evaluates in the `uni` setting. Refer to the `evl_attackability_det.py` script to see adjustments to passed arguments to evaluate in `vspec`, `spec` or `all` settings.



