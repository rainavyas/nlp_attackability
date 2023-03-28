python attack.py --out_dir experiments/attacks/val/bae --model_path experiments/trained_models/bert-base-uncased_sst_pretrainedTrue_seed1.th --model_name bert-base-uncased --data_name sst --num_classes 2 --attack_method bae --val --attack

python attack.py --out_dir experiments/attacks/val/bae --model_path experiments/trained_models/google-electra-base-discriminator_sst_pretrainedTrue_seed1.th --model_name google/electra-base-discriminator --data_name sst --num_classes 2 --attack_method bae --val --attack

python attack.py --out_dir experiments/attacks/val/bae --model_path experiments/trained_models/xlnet-base-cased_sst_pretrainedTrue_seed1.th --model_name xlnet-base-cased --data_name sst --num_classes 2 --attack_method bae --val --attack
