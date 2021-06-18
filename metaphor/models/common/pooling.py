import torch
from metaphor.models.common.tokenize import StandardTokenizer
import torch.nn as nn


class MeanPooler(nn.Module):
    def __init__(self, tokenizer: StandardTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, x: torch.Tensor):
        # N: num_sentences, M: max_sentence_length, D: embedding dimension
        # x has shape N x M x D, mask: N x M
        sentence_lengths = self.tokenizer.sentence_lengths  # N x 1
        return torch.sum(x * self.tokenizer.mask, dim=2).div_(sentence_lengths)
