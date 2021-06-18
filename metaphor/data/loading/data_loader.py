import os
import numpy as np
import pandas as pd
from metaphor.models.common.tokenize import StandardTokenizer
import torch
import torch.nn as nn
from typing import Tuple, List
import re


class Pheme:
    def __init__(
        self,
        file_path: str,
        tokenizer: StandardTokenizer,
        embedder: nn.Module,
        seed: int = 0,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
    ):
        if not (os.path.exists(file_path)):
            raise Exception(f"No such path: {file_path}")
        np.random.seed(seed)
        data = self._filter_dataset(pd.read_csv(file_path))
        data["text"] = self._preprocess_text(data["text"])
        data["veracity"] = self._process_labels(data["veracity"])
        print(data["text"].values)
        N = len(data["text"].values)
        tokenized_sentences = tokenizer([x for x in data["text"].values])
        embedder = embedder(tokenized_sentences)
        indxs = np.arange(N)
        indxs = np.random.permutation(indxs)
        l_split, r_split = int(train_size * N), int((train_size + val_size) * N)
        train_indxs = indxs[:l_split]
        val_indxs = indxs[l_split:r_split]
        test_indxs = indxs[r_split:]

        twids, embedding, labels = (
            data["tweet_id"],
            embedder(tokenized_sentences),
            data["veracity"],
        )
        self._train_loader = PhemeDataLoader(
            twids[train_indxs], embedding[train_indxs], labels[train_indxs]
        )
        self._val_loader = PhemeDataLoader(
            twids[val_indxs], embedding[val_indxs], labels[val_indxs]
        )
        self._test_loader = PhemeDataLoader(
            twids[test_indxs], embedding[test_indxs], labels[test_indxs]
        )

    def _filter_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        # remove unwanted datapoints
        return data.dropna()
        # return data.loc[[x is not None for x in data["veracity"]]]

    def _preprocess_text(self, tweets: List[str]) -> List[str]:
        # preprocess tweet data: remove URLs from tweets
        regex = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        return [re.sub(regex, "", text).strip() for text in tweets]

    def _process_labels(self, labels: List[str]) -> List[int]:
        # convert labels into integer classes
        mapping = {"false": 0, "unverified": 1, "true": 2}
        return [mapping[label] for label in labels]

    @property
    def train(self):
        return self._train_loader

    @property
    def val(self):
        return self._val_loader

    @property
    def test(self):
        return self._test_loader


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
