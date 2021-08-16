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
        batch_size: int = 128,
        splits: List[float] = [0.6, 0.2, 0.2],
        topic=None,
    ):
        if not (os.path.exists(file_path)):
            raise Exception(f"No such path: {file_path}")
        np.random.seed(0)
        data = self._filter_dataset(pd.read_csv(file_path))
        data["text"] = self._preprocess_text(data["text"])
        if topic is not None:
            topics = sorted(list(set(data["topic"])))
            data = data.loc[[topics.index(t) == topic for t in data["topic"]]]
        self.data = data
        labels = self.labels(data)
        N = len(data["text"].values)

        indxs = np.random.permutation(np.arange(N))
        l_split, r_split = int(splits[0] * N), int((splits[0] + splits[1]) * N)

        self.train_indxs = indxs[:l_split]
        self.val_indxs = indxs[l_split:r_split]
        self.test_indxs = indxs[r_split:]

        tokenized_sentences = tokenizer([x for x in data["text"].values])
        embedding = embedder(tokenized_sentences)

        train = PhemeDataset(
            self.train_indxs, embedding[self.train_indxs], labels[self.train_indxs]
        )
        val = PhemeDataset(
            self.val_indxs, embedding[self.val_indxs], labels[self.val_indxs]
        )
        test = PhemeDataset(
            self.test_indxs, embedding[self.test_indxs], labels[self.test_indxs]
        )

        self.train = torch.utils.data.DataLoader(train, batch_size=batch_size)
        self.val = torch.utils.data.DataLoader(val, batch_size=batch_size)
        self.test = torch.utils.data.DataLoader(test, batch_size=batch_size)

    def _filter_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        # remove unwanted datapoints
        return data.dropna()

    def _preprocess_text(self, tweets: List[str]) -> List[str]:
        # preprocess tweet data: remove URLs from tweets, remove hashtags and @
        regex = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        return [re.sub(regex, "", text).strip() for text in tweets]


class MisinformationPheme(Pheme):
    def __init__(
        self,
        file_path: str,
        tokenizer: StandardTokenizer,
        embedder: nn.Module,
        batch_size: int = 128,
        splits: List[float] = [0.6, 0.2, 0.2],
    ):
        super().__init__(file_path, tokenizer, embedder, batch_size, splits)

    def labels(self, data: pd.DataFrame) -> List[int]:
        # convert labels into integer classes
        labels = data["veracity"].values
        mapping = {"false": 0, "unverified": 1, "true": 2}
        return [mapping[label] for label in labels]


class PerTopicMisinformation:
    """
    A list of datasets per topic of the form [MisinformationPheme(),...,] where each only contains the tweets for a particular topic
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: StandardTokenizer,
        embedder: nn.Module,
        batch_size: int = 128,
        splits: List[float] = [0.6, 0.2, 0.2],
    ):
        n_topics = 9
        self.data = [
            Pheme(file_path, tokenizer, embedder, label=i) for i in range(n_topics)
        ]


class TopicPheme(Pheme):
    """
    Dataset of tweet to topic classification
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: StandardTokenizer,
        embedder: nn.Module,
        batch_size: int = 128,
        splits: List[float] = [0.6, 0.2, 0.2],
    ):
        super().__init__(file_path, tokenizer, embedder, batch_size, splits)

    def labels(self, data: pd.DataFrame) -> List[int]:
        topics = sorted(list(set(data["topic"])))
        return [topics.index(x) for x in data["topic"].values]


class PhemeDataset(torch.utils.data.Dataset):
    def __init__(
        self, tweet_ids: torch.Tensor, embedding: torch.Tensor, labels: torch.Tensor
    ):
        super().__init__()
        self.tweet_ids = tweet_ids
        self.embedding = embedding
        self.labels = labels

    def __len__(self):
        return self.embedding.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.tweet_ids[index], self.embedding[index]), self.labels[index]
