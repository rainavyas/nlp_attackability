import torch
from tqdm import tqdm

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
            print(min_perts)
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
            if cls._can_attack(sentence, y, attack, model):
                return size
        return 2.0

    
    @staticmethod
    def _construct_attack(model_wrapper, method, pert_size):
        if method == 'textfooler':
            cos_sim = 1-pert_size
            return TextFoolerJin2019.build(model_wrapper, min_cos_sim=cos_sim)

    @staticmethod
    def _can_attack(sentence, y, attack, model):
        '''
            Return True if sentence can be attacked: y_attack differs from y
        '''
        attack_result = attack.attack(sentence, y)
        updated_sentence = attack_result.perturbed_text()
        with torch.no_grad():
            logits = model.predict([updated_sentence])[0].squeeze()
            y_attack = torch.argmax(logits).detach().cpu().item()
        if y_attack != y:
            print(sentence)
            print('\n')
            print(updated_sentence)
            return True
        return False