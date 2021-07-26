import torch
import torch.nn as nn
import numpy as np
from typing import List
from metaphor.models.common.tokenize import Tokenizer


"""
An attack is formulated as a function on tokenized sentences and indices
In general, we would expect only to modify the val. and test sentences
in which the indices would contain the relevant val. and test. indices
"""


class Attack:
    pass


class KSynonymAttack(Attack):
    """Replace k words in the sentence with a synonym such that the likelihood
    of misclassification is maximised"""

    def __init__(
        self,
        k: int,
        embedding,
        tokenizer: Tokenizer,
        model: nn.Module,
        attempts: int = 5,
        N: int = 5,
        batch_size: int = 128,
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
        self.embedding = embedding
        self.model = model
        self.attempts = attempts

    def attack(self, sentences: List[str], labels: List[int]) -> List[str]:
        """
        Takes a list of sentences of the form: ["The Quick brown fox", "jumps over", "the lazy dog", ...]
        and 'attacks' them in batches by selecting synonyms from the embedding
        Returns predictions and the attacked sentences
        Logs to a file: all the attacked sentences, predictions by the model on each and the true labels
        """
        predictions = torch.zeros((self.attempts, len(sentences)))
        attacked_sentences = []
        for start in range(0, len(sentences), self.batch_size):
            end = min(len(sentences), start + self.batch_size)
            # indices of words to substitute
            for sentence in sentences[start:end]:
                attacked_sen = []
                for i in range(self.attempts):
                    indxs = np.random.choice(len(sentence), size=self.k)
                    for j in indxs:
                        new_sent = sentence.copy()
                        synonyms = [
                            x[0]
                            for x in self.embedding.most_similar(sentence[j])[: self.N]
                        ]
                        new_sent[j] = np.random.choice(synonyms)
                    attacked_sen.append(new_sent)
                attacked_sentences.append(attacked_sen)
        print(attacked_sentences)
        # embed the new sentence and evaluate on model
        # if there's a change in classication then break
        # tokens = self.tokenizer([sentence, new_sent])
        # embedding = self.embedding(self.tokenizer)
        # ys = self.model(embedding)
        # print(ys)
