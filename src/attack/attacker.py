import torch
from tqdm import tqdm
import numpy as np

from .model_wrapper import PyTorchModelWrapper
from .redefined_textattack_models import TextFoolerJin2019

class Attacker():

    @classmethod
    def get_all_pert_sizes(cls, sentences, model, method='textfooler', sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        '''
            Get minimum perturbation size (cosine distance) to successfully change prediction of a sample
        '''
        min_perts = []
        for sentence in tqdm(sentences):
            min_perts.append(cls.get_pert_size(sentence, model, method, sizes))
            # print(min_perts)
        return min_perts

    @classmethod
    def get_pert_size(cls, sentence, model, method='textfooler', sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        '''
            Find smallest perturbation to change output class
            sizes: pert sizes are the cosine distances (1-cos_sim) in an embedding space

            If all sample sizes fail, return maximum cosine_distance = 2.0
        '''
        # get original predicted class
        with torch.no_grad():
            logits = model.predict([sentence])[0].squeeze()
            y = torch.argmax(logits).detach().cpu().item()

        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
        for size in sizes:
            attack =  cls._construct_attack(model_wrapper, method, size)
            if cls._can_attack(sentence, y, attack):
                return size
        return 2.0

    
    @staticmethod
    def _construct_attack(model_wrapper, method, pert_size):
        if method == 'textfooler':
            cos_sim = 1-pert_size
            attack = TextFoolerJin2019.build(model_wrapper, min_cos_sim=cos_sim)
        return attack

    @staticmethod
    def _can_attack(sentence, y, attack):
        '''
            Return True if sentence can be attacked: y_attack differs from y
        '''
        attack_result = attack.attack(sentence, y)
        out = attack_result.goal_function_result_str()
        if 'FAILED' in out:
            return False
        return True

    @staticmethod
    def attack_frac_sweep(perts, threshs=[0.001, 0.101, 0.201, 0.301, 0.401, 0.501, 0.601]):
        '''
        Return fraction of attackable samples at each perturbation size threshold
        '''
        size = len(perts)
        frac_attackable = []
        for t in threshs:
            num_att = len([p for p in perts if p<=t])
            frac_attackable.append(num_att/size)
        return threshs, frac_attackable

    @staticmethod
    def attack_frac_sweep_all(perts_all, threshs=[0.001, 0.101, 0.201, 0.301, 0.401, 0.501, 0.601]):
        '''
        Return fraction of attackable samples (over all models) at each perturbation size threshold
        '''
        size = len(perts_all[0])
        frac_attackable = []
        for t in threshs:
            num_att = 0
            for sample in zip(*perts_all):
                smaller = True
                for pert in sample:
                    if pert > t:
                        smaller = False
                        break
                if smaller:
                    num_att+=1
            frac_attackable.append(num_att/size)
        return threshs, frac_attackable
