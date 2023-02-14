from textattack import Attack
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding

from textattack.attack_recipes.attack_recipe import AttackRecipe

from textattack.transformations import WordSwapMaskedLM

import math


class TextFoolerJin2019(AttackRecipe):
    """Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019).

    Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment.

    https://arxiv.org/abs/1907.11932
    """

    @staticmethod
    def build(model_wrapper, min_cos_sim=0.5):
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        #
        # Minimum word embedding cosine similarity.
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=min_cos_sim))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #
        # Universal Sentence Encoder with a minimum angular similarity of threshold.
        #
        use_constraint = UniversalSentenceEncoder(
            threshold=1-(math.acos(min_cos_sim)/math.pi), # angular_sim = 1- (arccos(cos_sim)/pi)
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)

class BAEGarg2019(AttackRecipe):
    """Siddhant Garg and Goutham Ramakrishnan, 2019.

    BAE: BERT-based Adversarial Examples for Text Classification.

    https://arxiv.org/pdf/2004.01970

    This is "attack mode" 1 from the paper, BAE-R, word replacement.

    We present 4 attack modes for BAE based on the
        R and I operations, where for each token t in S:
        • BAE-R: Replace token t (See Algorithm 1)
        • BAE-I: Insert a token to the left or right of t
        • BAE-R/I: Either replace token t or insert a
        token to the left or right of t
        • BAE-R+I: First replace token t, then insert a
        token to the left or right of t
    """

    @staticmethod
    def build(model_wrapper, min_cos_sim=0.5):
        #
        transformation = WordSwapMaskedLM(
            method="bae", max_candidates=50, min_confidence=0.0
        )
        #
        # Don't modify the same word twice or stopwords.
        #
        constraints = [RepeatModification(), StopwordModification()]

        # For the R operations we add an additional check for
        # grammatical correctness of the generated adversarial example by filtering
        # out predicted tokens that do not form the same part of speech (POS) as the
        # original token t_i in the sentence.
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

        # "To ensure semantic similarity on introducing perturbations in the input
        # text, we filter the set of top-K masked tokens (K is a pre-defined
        # constant) predicted by BERT-MLM using a Universal Sentence Encoder (USE)
        # (Cer et al., 2018)-based sentence similarity scorer."
        #
        use_constraint = UniversalSentenceEncoder(
            threshold=1-(math.acos(min_cos_sim)/math.pi), # angular_sim = 1- (arccos(cos_sim)/pi)
            metric="angular",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification.
        #
        goal_function = UntargetedClassification(model_wrapper)

        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)