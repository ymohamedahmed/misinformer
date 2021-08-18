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
        label_fun,
        batch_size: int = 128,
        splits: List[float] = [0.6, 0.2, 0.2],
    ):
        if not (os.path.exists(file_path)):
            raise Exception(f"No such path: {file_path}")
        np.random.seed(0)
        data = self._filter_dataset(pd.read_csv(file_path))
        data["text"] = self._preprocess_text(data["text"])
        self.data = data
        labels = label_fun(data)
        N = len(data["text"].values)

        indxs = np.random.permutation(np.arange(N))
        l_split, r_split = int(splits[0] * N), int((splits[0] + splits[1]) * N)

        self.train_indxs = indxs[:l_split]
        self.val_indxs = indxs[l_split:r_split]
        self.test_indxs = indxs[r_split:]

        tokenized_sentences = tokenizer([x for x in data["text"].values])
        embedding = embedder(tokenized_sentences)

        self.train = torch.utils.data.DataLoader(
            PhemeDataset(
                self.train_indxs,
                embedding[self.train_indxs],
                labels[self.train_indxs],
            ),
            batch_size=batch_size,
        )
        self.val = torch.utils.data.DataLoader(
            PhemeDataset(
                self.val_indxs, embedding[self.val_indxs], labels[self.val_indxs]
            ),
            batch_size=batch_size,
        )
        self.test = torch.utils.data.DataLoader(
            PhemeDataset(
                self.test_indxs, embedding[self.test_indxs], labels[self.test_indxs]
            ),
            batch_size=batch_size,
        )
        self.embedding = embedding
        self.labels = labels
        self.batch_size = batch_size

    def per_topic(self):
        """Return data loaders separated by each topic"""
        topics = sorted(list(set(self.data["topic"])))
        # list of indices in each topic
        topic_indxs = [
            [
                i
                for i in range(len(self.data["topic"]))
                if topics.index(self.data["topic"].values[i]) == j
            ]
            for j in range(len(topics))
        ]
        # rewrite the indices in each fold per-topic i.e. [[# the train indices in the 0th topic], ...]

        def filter_indices(t):
            return [[y for y in x if y in t] for x in topic_indxs]

        indices = [
            filter_indices(t)
            for t in [self.train_indxs, self.val_indxs, self.test_indxs]
        ]

        for i, ind in enumerate(indices):
            for j, top in enumerate(ind):
                print(
                    f'{"Train" if i == 0 else "Val" if i == 1 else "Test"} Topic: {j} Length: {len(top)}'
                )

        # indices = [[np.array(t) for t in ind] for ind in indices]

        return [
            [
                torch.utils.data.DataLoader(
                    PhemeDataset(t, self.embedding[t], self.labels[t]),
                    batch_size=self.batch_size,
                )
                for t in ind
            ]
            for ind in indices
        ]

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
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            embedder=embedder,
            batch_size=batch_size,
            splits=splits,
            label_fun=self.labels,
        )

    def labels(self, data: pd.DataFrame) -> List[int]:
        # convert labels into integer classes
        labels = data["veracity"].values
        mapping = {"false": 0, "unverified": 1, "true": 2}
        return np.array([mapping[label] for label in labels])


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
            Pheme(file_path, tokenizer, embedder, topic=i, label_fun=self.labels)
            for i in range(n_topics)
        ]

    def labels(self, data: pd.DataFrame) -> List[int]:
        # convert labels into integer classes
        labels = data["veracity"].values
        mapping = {"false": 0, "unverified": 1, "true": 2}
        return np.array([mapping[label] for label in labels])


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
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            embedder=embedder,
            batch_size=batch_size,
            splits=splits,
            label_fun=self.labels,
        )

    def labels(self, data: pd.DataFrame) -> List[int]:
        topics = sorted(list(set(data["topic"])))
        return np.array([topics.index(x) for x in data["topic"].values])


class PhemeDataset(torch.utils.data.Dataset):
    def __init__(
        self, tweet_ids: torch.Tensor, embedding: torch.Tensor, labels: torch.Tensor
    ):
        super().__init__()
        print(embedding.shape)
        self.tweet_ids = tweet_ids
        self.embedding = embedding
        self.labels = labels

    def __len__(self):
        return self.embedding.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            z = (self.tweet_ids[index], self.embedding[index], self.labels[index])
        except IndexError as e:
            print(e)
            print(index)
            print(self.embedding.shape)
            print(self.embedding.shape[0])
            print(self.tweet_ids)
            print(self.embedding)
            print(self.labels)

        return (self.tweet_ids[index], self.embedding[index]), self.labels[index]
