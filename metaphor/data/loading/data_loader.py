import os
import numpy as np
import pandas as pd
from metaphor.models.common.tokenize import Tokenizer
import torch
from typing import Tuple, List


class Pheme:
    def __init__(
        self,
        file_path: str,
        tokenizer: Tokenizer,
        seed: int = 0,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
    ):
        if not (os.path.exists(file_path)):
            raise Exception(f"No such path: {file_path}")
        np.random.seed(seed)
        data = pd.read_csv(file_path)
        N = len(data["text"])
        tokenized_sentences = tokenizer(data["text"])
        indxs = np.arange(N)
        indxs = np.random.permutation(indxs)
        l_split, r_split = int(train_size * N), int((train_size + val_size) * N)
        train_indxs = indxs[:l_split]
        val_indxs = indxs[l_split:r_split]
        test_indxs = indxs[r_split:]

        self._train_loader = PhemeDataLoader()

    def _preprocess_text(self, tweets: List[str]) -> List[str]:
        # preprocess tweet data:
        pass

    def _process_labels(self, labels: List[str]) -> List[int]:
        # convert labels into integer classes
        pass


class PhemeDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self, tweet_ids: torch.Tensor, embedding: torch.Tensor, labels: torch.Tensor
    ):
        self.tweet_ids = tweet_ids
        self.embedding = embedding
        self.labels = labels

    def __len__(self):
        return self.embedding.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.tweet_ids[index], self.embedding[index]), self.labels[index]
