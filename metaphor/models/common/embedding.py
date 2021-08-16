from metaphor.models.common.pooling import MeanPooler
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np
import gensim
import gensim.downloader as api
from metaphor.models.common.tokenize import (
    StandardTokenizer,
    CustomBertTokenizer,
    Tokenizer,
)
import transformers
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sentence - Glove/Bert -> Tensor of embeddings - CNN/RNN/MeanPool -> Single vector
# e.g. nn.Sequential([Bert(), RNN()]) or nn.Sequential(Glove])


class Bert(nn.Module):
    def __init__(
        self,
        tokenizer: CustomBertTokenizer,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        batch_size: int = 128,
    ):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def forward(self, x: torch.IntTensor):
        x = x.to(self.device)
        embeddings = torch.zeros(x.shape[0], x.shape[1], self.model.config.hidden_size)
        mask = self.tokenizer.mask.to(self.device)
        for start in range(0, x.shape[0], self.batch_size):
            with torch.no_grad():
                end = min(x.shape[0], start + self.batch_size)
                embeddings[start:end] = self.model(
                    input_ids=x[start:end],
                    attention_mask=mask[start:end],
                    output_attentions=False,
                ).last_hidden_state
        return embeddings


class Glove(nn.Module):
    def __init__(
        self,
        tokenizer: StandardTokenizer,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._embed = None
        print("Downloading glove twitter vectors")
        self.model = api.load("glove-twitter-200")

    def _construct_embedding(self):
        embedding_dim = self.model.vector_size
        OOV = 0
        # randomly initialise OOV tokens between -1 and 1
        weights = np.random.uniform(
            low=-1, high=1, size=(len(self._tokenizer.dictionary), embedding_dim)
        )

        mean_embedding = np.zeros(embedding_dim)
        vec_count = 1

        print("Begin computing embedding")
        # represent unknown tokens with the mean of the embeddings
        for word, token in self._tokenizer.dictionary.token2id.items():
            # print(f"{word} {token}")
            if word == "<PAD>":
                weights[token, :] = np.zeros(embedding_dim)
            elif word in self.model.key_to_index:
                vec = self.model[word]
                vec_count += 1
                mean_embedding += vec
                weights[token, :] = vec
            else:
                OOV += 1

        mean_embedding = mean_embedding / vec_count
        for word, token in self._tokenizer.dictionary.token2id.items():
            if not (word in self.model.key_to_index):
                weights[token, :] = mean_embedding

        # print number out of vocab
        print(f"done. Embedding dim: {embedding_dim}. Number of OOV tokens: {OOV}")
        self._embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights))

    def forward(self, x: torch.Tensor):
        # x is the tokens representing the sentence
        if self._embed is None:
            self._construct_embedding()
        return self._embed(x)


class RNN(nn.Module):
    def __init__(
        self, tokenizer: StandardTokenizer, hidden_dim: int, embedding_size: int
    ):
        super().__init__()
        self.tokenizer = tokenizer
        # self.dictionary = dictionary
        self.hidden_dim = hidden_dim // 2  # assuming bidirectional
        # self.padding_token = self.dictionary["PAD"]

        # self.text_emb_size = embedding_weights.shape[-1]

        # RNN
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x, ind):
        """Params:
        - x (torch.LongTensor): tensor of embeddings shape (B x max_sequence_length x emb_dim)
        Returns:
        - text_embedding (torch.FloatTensor): embedded sequence (B x emb_dim)
        """
        # process data
        sent_lens = self.tokenizer.sentence_lengths[ind]
        packed = pack_padded_sequence(
            x, sent_lens, batch_first=True, enforce_sorted=False
        )
        out, hidden = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        seq_len_indices = [(length - 1).item() for length in sent_lens]
        batch_indices = [i for i in range(x.shape[0])]
        rnn_out_forward = out[
            batch_indices, seq_len_indices, : self.hidden_dim
        ]  # last state of forward (not padded)
        rnn_out_backward = out[
            :, 0, self.hidden_dim :
        ]  # last state of backward (= first timestep)
        seq_embed = torch.cat(
            (rnn_out_forward, rnn_out_backward), -1
        )  # (B*N*K, rnn_hid_dim*2)
        return seq_embed


class CNN(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        conv_channels: List[int],
        output_dim: int,
        kernel_sizes: List[int],
        stride=1,
        padding=1,
    ):
        super().__init__()
        # layers = [nn.BatchNorm1d(conv_channels[0])]
        layers = []
        for i in range(len(conv_channels) - 1):
            layers.append(
                nn.Conv1d(
                    conv_channels[i],
                    conv_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.Dropout(p=0.1))
            # layers.append(nn.BatchNorm1d(conv_channels[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=kernel_sizes[i], stride=1))
        layers.pop()
        layers.append(nn.AdaptiveMaxPool1d(output_size=output_dim))

        layers.append(nn.Flatten())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, ind) -> torch.Tensor:
        s = x.shape
        out = self.model(x.reshape(s[0], s[2], s[1]))
        return out


class MLP(nn.Module):
    def __init__(self, layer_dimensions: List[int]):
        super().__init__()
        # mlp_layers = [conv_channels[-1]*28*28] + mlp_layers
        layers = []
        for i in range(len(layer_dimensions) - 1):
            layers.append(nn.Linear(layer_dimensions[i], layer_dimensions[i + 1]))
            if i < len(layer_dimensions) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ExpertMixture(nn.Module):
    def __init__(self, n_topics: int, model_args, model, topic_selector):
        self.topic_selector = topic_selector
        self.models = [model(**model_args) for _ in range(n_topics)]
        self.n_topics = n_topics

    def forward(self, x):
        topics = self.topic_selector(x).argmax(dim=1)
        preds = torch.zeros(x.shape[0], 3)
        for i in range(topics):
            preds[topics == i] = self.models[i](x[topics == i])
        return preds

    def fit(self, trainer, topic_classification_loader, misinformation_loader):
        trainer.fit(
            self.topic_selector,
            topic_classification_loader.train,
            topic_classification_loader.val,
        )
        for i in range(self.n_topics):
            # need to fit only to particular topic
            trainer.fit(
                self.models[i],
                misinformation_loader.data[i].train,
                misinformation_loader.data[i].val,
            )
