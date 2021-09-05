import os
import numpy as np
import pandas as pd
from metaphor.models.common.tokenize import StandardTokenizer
import torch
import torch.nn as nn
from typing import Tuple, List
from itertools import chain, combinations
import re


class Pheme:
    def __init__(
        self,
        file_path: str,
        tokenizer: StandardTokenizer,
        embedder: nn.Module,
        label_fun,
        batch_size: int = 128,
        seed=0,
        splits: List[float] = [0.6, 0.2, 0.2],
        indxs=None,
        augmented_sentences=None,
    ):
        # TODO add support for augmentation

        if not (os.path.exists(file_path)):
            raise Exception(f"No such path: {file_path}")
        np.random.seed(seed)
        data = self._filter_dataset(pd.read_csv(file_path))
        data["text"] = self._preprocess_text(data["text"])
        self.data = data
        labels = label_fun(data)
        N = len(data["text"].values)

        if indxs is None:
            indxs = np.random.permutation(np.arange(N))
            l_split, r_split = int(splits[0] * N), int((splits[0] + splits[1]) * N)
            self.train_indxs = indxs[:l_split]
            self.val_indxs = indxs[l_split:r_split]
            self.test_indxs = indxs[r_split:]
        else:
            self.train_indxs = indxs[0]
            self.val_indxs = indxs[1]
            self.test_indxs = indxs[2]

        sentences = [x for x in data["text"].values]
        if augmented_sentences is not None:
            sentences = sentences + augmented_sentences

        tokenized_sentences = tokenizer(sentences)
        embedding = embedder(tokenized_sentences)

        augmentation = None
        if augmented_sentences is not None:
            # embedding has shape (N+len(augmented_sentences))*max_sentence_len*embedding_dim
            # We want it reshaped into max_sentence_len * len(augmented_sentences) * embedding_dim
            x, y, z = embedding.shape
            augmentation = embedding[N:].reshape((y, x - N, z))

        self.train = torch.utils.data.DataLoader(
            PhemeDataset(
                self.train_indxs,
                embedding[self.train_indxs],
                labels[self.train_indxs],
                augmentation=augmentation,
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

    def _topic_indxs(self):
        topics = sorted(list(set(self.data["topic"])))
        # list of indices in each topic
        # i.e. [[indices for 0th topic], [indices for 1st topic],...]
        return [
            [
                i
                for i in range(len(self.data["topic"]))
                if topics.index(self.data["topic"].values[i]) == j
            ]
            for j in range(len(topics))
        ]

    def per_topic(self):
        """Return data loaders separated by each topic"""

        # rewrite the indices in each fold per-topic i.e. [[ the train indices in the 0th topic], ...]
        topic_indxs = self._topic_indxs()

        def filter_indices(t):
            return [[y for y in x if y in t] for x in topic_indxs]

        indices = [
            filter_indices(t)
            for t in [self.train_indxs, self.val_indxs, self.test_indxs]
        ]

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
        seed=0,
        batch_size: int = 128,
        augmented_sentences=None,
        splits: List[float] = [0.6, 0.2, 0.2],
    ):
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            embedder=embedder,
            batch_size=batch_size,
            splits=splits,
            label_fun=self.labels,
            augmented_sentences=augmented_sentences,
            seed=seed,
        )

    def labels(self, data: pd.DataFrame) -> List[int]:
        # convert labels into integer classes
        labels = data["veracity"].values
        mapping = {"false": 0, "unverified": 1, "true": 2}
        return np.array([mapping[label] for label in labels])


class HardPheme(Pheme):
    """
    Withhold a topic/multiples topics for the test set from the model

    """

    def __init__(
        self,
        file_path: str,
        tokenizer: StandardTokenizer,
        embedder: nn.Module,
        seed,
        batch_size: int = 128,
        splits: List[float] = [0.6, 0.2, 0.2],
    ):
        np.random.seed(seed)
        # select withheld topics randomly
        data = self._filter_dataset(pd.read_csv(file_path))
        data["text"] = self._preprocess_text(data["text"])
        witholdable_topics = self._topic_groups(data)
        witheld_topics = np.random.choice(witholdable_topics)
        test_indxs = [x in witheld_topics for x in data["topic"]]
        test_indxs = np.arange(len(data))[test_indxs]

        # select which topics to withold
        # split the remaining points into train and val as normal i.e 3:1 split from the remainder
        remaining = [i for i in range(len(data)) if i not in test_indxs]
        indxs = np.random.permutation(remaining)
        split = int(0.75 * len(indxs))
        train_indxs = indxs[:split]
        val_indxs = indxs[split:]

        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            embedder=embedder,
            batch_size=batch_size,
            splits=splits,
            label_fun=self.labels,
            indxs=[train_indxs, val_indxs, test_indxs],
        )

    from itertools import chain, combinations

    def _powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def _topic_groups(self, data):
        # returns the groups of topics with enough datapoints to make a large enough test
        # considered to be any group covering between 15% and 25% of the dataset
        gr = 100 * data.groupby("topic").count()["veracity"] / len(data)
        pset = list(self._powerset(gr.keys()))
        valid_topics = []
        for combo in pset:
            coverage = 0
            for x in combo:
                coverage += gr[x]
            if coverage > 15 and coverage < 25:
                valid_topics.append(combo)

        return valid_topics

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
        self,
        tweet_ids: torch.Tensor,
        embedding: torch.Tensor,
        labels: torch.Tensor,
        augmentation: torch.Tensor = None,
        augment_prob=0.1,
    ):
        super().__init__()
        self.tweet_ids = tweet_ids
        self.embedding = embedding
        self.labels = labels
        self._aug = augmentation
        self._aug_p = augment_prob

    def __len__(self):
        return self.embedding.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._aug is not None:
            if np.random.uniform() < self._aug_p:
                rand = np.random.randint(low=0, high=self._aug.shape[1])
                return (
                    self.tweet_ids[index],
                    self._aug[index][rand],
                ), self.labels[index]

        return (self.tweet_ids[index], self.embedding[index]), self.labels[index]
