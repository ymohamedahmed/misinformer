from typing import List
from gensim import corpora
from gensim.utils import tokenize
import numpy as np
from itertools import chain
import transformers
import torch


class PreTokenizationError(Exception):
    pass


class Tokenizer:
    pass


class CustomBertTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            sentences,
            return_token_type_ids=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        self.mask = tokens["attention_mask"]
        return tokens["input_ids"]

    @property
    def max_length(self) -> int:
        return self.mask.shape[1]


    @property
    def sentence_lengths(self) -> torch.Tensor:
        # return self._sentence_lengths
    

    @property
    def mask(self) -> torch.Tensor:
        pass


class StandardTokenizer(Tokenizer):
    def __init__(self):
        self._dictionary = None
        self._sentence_lengths = None
        self._padded_sentences = None

    def _check_tokenizer_called(self):
        if (
            self._sentence_lengths is None
            or self._dictionary is None
            or self._padded_sentences is None
        ):
            raise PreTokenizationError(
                "Sentence lengths and token2id dictionary are undefined before tokenization"
            )

    @property
    def sentence_lengths(self) -> torch.Tensor:
        self._check_tokenizer_called()
        return self._sentence_lengths

    @property
    def dictionary(self) -> corpora.Dictionary:
        self._check_tokenizer_called()
        return self._dictionary

    @property
    def padded_sentences(self) -> List[str]:
        self._check_tokenizer_called()
        return self._padded_sentences

    @property
    def mask(self) -> List[str]:
        self._check_tokenizer_called()
        return self._mask

    @property
    def max_length(self) -> int:
        return self._max_length

    def _padding(self, sentences: List[str], lengths: List[int]) -> List[str]:
        padding = []
        self._max_length = max(lengths)
        for i in range(len(sentences)):
            diff = self.max_length - lengths[i]
            padding.append(["<PAD>" for _ in range(diff)])
        return padding

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        lengths = [sum([1 for w in tokenize(s)]) for s in sentences]
        dictionary = corpora.Dictionary(
            [tokenize(d.lower()) for i, d in enumerate(sentences)]
        )
        sentences = [s.lower() for s in sentences]
        dictionary.patch_with_special_tokens({"<PAD>": 0})

        padding = self._padding(sentences, lengths)
        padded_sentences = [
            " ".join([x for x in tokenize(d.lower())] + padding[i])
            for (i, d) in enumerate(sentences)
        ]
        tokenized_sentences = torch.Tensor(
            [
                [dictionary.token2id[z] for z in padded_sentences[i].split(" ")]
                for i, d in enumerate(sentences)
            ]
        )

        self._dictionary = dictionary
        self._padded_sentences = padded_sentences
        self._sentence_lengths = torch.Tensor(lengths)
        self._mask = tokenized_sentences != self.dictionary.token2id["<PAD>"]

        return tokenized_sentences
