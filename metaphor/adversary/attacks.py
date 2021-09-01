import torch
import torch.nn as nn
import numpy as np
from typing import List
from metaphor.models.common.tokenize import Tokenizer
import gensim.downloader as api
from gensim.utils import tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

"""
An attack is formulated as a function on tokenized sentences and indices
In general, we would expect only to modify the val. and test sentences
in which the indices would contain the relevant val. and test. indices
"""


class Attack:
    def __init__(self, path):
        pass


class ConcatenationAttack(Attack):
    def __init__(
        self,
        attack: str,
    ):
        self.attack = attack

    def attack(self, sentences):
        return [x + " " + self.attack for x in sentences]


class Misinformer(Attack):
# class LimeConcatenationAttack(ConcatenationAttack):
#     def __init__(self, lime_path: str, N):
#         """
#         lime_path: the path to the global explainers
#         N: the top N explainers to use in the concatenation attack
#         """

#         super().__init__(attack=..)


class ParaphraseAttack(Attack):
    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        attempts: int = 5,
        path: str = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.path = path
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.device = device
        self.model.to(device)
        self.attempts = attempts

    def attack(self, sentences: List[str]):
        if self.path is not None:
            with open(self.path) as f:
                lines = f.readlines()
                return [x.strip() for x in lines]
        attacked_sentences = []
        for sentence in sentences:
            encoding = self.tokenizer.encode_plus(
                sentence, pad_to_max_length=True, return_tensors="pt"
            )
            input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding[
                "attention_mask"
            ].to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=256,
                do_sample=True,
                top_k=200,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=self.attempts,
            )

            lines = []
            for output in outputs:
                line = self.tokenizer.decode(
                    output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                attacked_sentences.append(line)

        return attacked_sentences


class KSynonymAttack(Attack):
    """Replace k words in the sentence with a synonym such that the likelihood
    of misclassification is maximised"""

    def __init__(
        self,
        k: int,
        attempts: int = 5,
        N: int = 3,
        batch_size: int = 128,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        path: str = None,
    ):
        """
        k: the number of words to substitute
        embedding: either Glove or BERT
        model: trained model
        max_length: length of the longest sentence
        attempts: maximum number of attacks (i.e. number of different substitutions attempted)
        N: number of nearest neighbours to consider when picking synonym in embedding space
        e.g. KSynonymAttack(k=5, embedding=Glove, tokenizer=StandardTokenizer, model=...,)
        """
        np.random.seed(0)
        self.k = k
        self.attempts = attempts
        self.N = N
        self.synonym_model = api.load("glove-wiki-gigaword-300")
        self.device = device
        self.path = path

    def attack(self, sentences: List[str]):
        if self.path is not None:
            with open(self.path) as f:
                lines = f.readlines()
                return [x.strip() for x in lines]

        attacked_sentences = []
        self.num_sentences = len(sentences)
        # indices of words to substitute
        for sentence in tqdm(sentences):
            sentence = [x for x in tokenize(sentence.lower())]
            for i in range(self.attempts):
                indxs = np.random.choice(len(sentence), size=self.k)
                new_sent = sentence.copy()
                for j in indxs:
                    # if the words isn't in the embedding space don't replace it
                    synonyms = [sentence[j]]
                    try:
                        self.synonym_model.get_index(sentence[j])
                        synonyms = [
                            x[0]
                            for x in self.synonym_model.most_similar(sentence[j])[
                                : self.N
                            ]
                        ]
                    except KeyError:
                        pass

                    new_sent[j] = np.random.choice(synonyms)
                attacked_sentences.append(" ".join(new_sent))
        return attacked_sentences
