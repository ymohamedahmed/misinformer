import torch
from metaphor.models.common.tokenize import StandardTokenizer
import torch.nn as nn
import numpy as np


class MeanPooler(nn.Module):
    def __init__(self, tokenizer: StandardTokenizer):
        super().__init__()
        self.device = (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),)
        self.tokenizer = tokenizer

    def forward(self, x: torch.Tensor, indxs: torch.Tensor):
        # N: num_sentences, M: max_sentence_length, D: embedding dimension
        # x has shape N x M x D, mask: N x M
        sentence_lengths = self.tokenizer.sentence_lengths[indxs].to(
            self.device
        )  # N x 1
        return torch.sum(
            x * self.tokenizer.mask[indxs].unsqueeze(2).to(self.device), dim=2
        ).div_(sentence_lengths)


class MisinformationModel(nn.Module):
    def __init__(self, aggregator: nn.Module, classifier: nn.Module):
        # aggregator takes a series of embeddings and produces a single vector
        # classifier uses this to produce a decision
        super().__init__()
        self._agg = aggregator
        self._clfr = classifier

    def forward(self, x: torch.Tensor, indxs: np.array):
        x = self._agg(x, indxs)
        return self._clfr(x)
