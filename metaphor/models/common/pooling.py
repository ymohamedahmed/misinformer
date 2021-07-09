import torch
from metaphor.models.common.tokenize import StandardTokenizer
import torch.nn as nn
import numpy as np


class MeanPooler(nn.Module):
    def __init__(self, tokenizer: StandardTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, x: torch.Tensor, indxs: np.array):
        # N: num_sentences, M: max_sentence_length, D: embedding dimension
        # x has shape N x M x D, mask: N x M
        sentence_lengths = self.tokenizer.sentence_lengths[indxs]  # N x 1
        return torch.sum(x * self.tokenizer.mask[indxs], dim=2).div_(sentence_lengths)
