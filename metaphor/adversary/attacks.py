import torch
import torch.nn as nn
import numpy as np
from typing import List
from metaphor.models.common.tokenize import Tokenizer
import gensim.downloader as api
from gensim.utils import tokenize

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
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
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
        self.batch_size = batch_size
        self.N = N
        self.synonym_model = api.load("glove-wiki-gigaword-300")
        self.tokenizer = tokenizer
        self.device = device

    def attack(self, sentences: List[str]) -> List[str]:
        """
        Takes a list of sentences of the form: ["The Quick brown fox", "jumps over", "the lazy dog", ...]
        and 'attacks' them in batches by selecting synonyms from the embedding
        Returns predictions and the attacked sentences
        Logs to a file: all the attacked sentences, predictions by the model on each and the true labels
        """
        predictions = torch.zeros((self.attempts * len(sentences)))
        attacked_sentences = []
        # indices of words to substitute
        for sentence in sentences:
            sentence = [x for x in tokenize(sentence.lower())]
            for i in range(self.attempts):
                indxs = np.random.choice(len(sentence), size=self.k)
                for j in indxs:
                    new_sent = sentence.copy()
                    synonyms = [
                        x[0]
                        for x in self.synonym_model.most_similar(sentence[j])[: self.N]
                    ]
                    new_sent[j] = np.random.choice(synonyms)
                attacked_sentences.append(" ".join(new_sent))

        tokenized_sentences = self.tokenizer(attacked_sentences)
        embedding = self.embedding(tokenized_sentences).to(self.device)

        for start in range(0, len(sentences) * self.attempts, self.batch_size):
            end = min(len(sentences), start + self.batch_size)
            y = self.model(embedding[start:end])
            predictions[start:end] = torch.argmax(y, dim=1)

        predictions = predictions.reshape((self.attempts, len(sentences)))
        return attacked_sentences, predictions

        # embed the new sentence and evaluate on model
        # if there's a change in classication then break
        # tokens = self.tokenizer([sentence, new_sent])
        # embedding = self.embedding(self.tokenizer)
        # ys = self.model(embedding)
        # print(ys)
