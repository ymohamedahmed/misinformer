from metaphor.models.common.pooling import MeanPooler
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np
import gensim
import gensim.downloader as api
from metaphor.models.common.tokenize import StandardTokenizer, CustomBertTokenizer
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
        print("Downloading glove gigaword")
        word_model = api.load("glove-wiki-gigaword-300")
        embedding_dim = word_model.vector_size

        OOV = 0
        # randomly initialise OOV tokens between -1 and 1
        weights = np.random.uniform(
            low=-1, high=1, size=(len(tokenizer.dictionary), embedding_dim)
        )

        mean_embedding = np.zeros(embedding_dim)
        vec_count = 1

        print("Begin computing embedding")
        # represent unknown tokens with the mean of the embeddings
        for word, token in tokenizer.dictionary.token2id.items():
            print(f"{word} {token}")
            if word == "<PAD>":
                weights[token, :] = np.zeros(embedding_dim)
            elif word in word_model.key_to_index:
                vec = word_model[word]
                vec_count += 1
                mean_embedding += vec
                weights[token, :] = vec
            else:
                OOV += 1

        mean_embedding = mean_embedding / vec_count
        for word, token in tokenizer.dictionary.token2id.items():
            if not (word in word_model.key_to_index):
                print(f"{word} {token}")
                weights[token, :] = mean_embedding

        # print number out of vocab
        print(f"done. Embedding dim: {embedding_dim}. Number of OOV tokens: {OOV}")
        self._embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights))

    def forward(self, x: torch.Tensor):
        # x is the tokens representing the sentence
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
        B, max_seq_len, embedding_dim = x.shape
        # flatten batch
        x_flat = x.view(-1, max_seq_len)  # (B*emb_dim x max_seq_len)

        # padding_masks
        # padding_mask = torch.where(x_flat != self.padding_token, 1, 0)
        # seq_lens = torch.sum(padding_mask, dim=-1).cpu()  # (B*N*K)

        # embed
        # text_embedding = self.embed(x_flat)  # (B*N*K x max_seq_len x emb_dim)

        # feed through RNN
        text_embedding_packed = pack_padded_sequence(
            x_flat,
            self.tokenizer.sentence_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_out_packed, _ = self.rnn(text_embedding_packed)
        rnn_out, _ = pad_packed_sequence(
            rnn_out_packed, batch_first=True
        )  # (B*N*K, max_seq_len, rnn_hid_dim*2)

        # concat forward and backward results (takes output states)
        seq_len_indices = [
            length - 1 for length in self.tokenizer.sentence_lengths[ind]
        ]
        batch_indices = [i for i in range(B)]
        rnn_out_forward = rnn_out[
            batch_indices, seq_len_indices, : self.hidden_dim
        ]  # last state of forward (not padded)
        rnn_out_backward = rnn_out[
            :, 0, self.hidden_dim :
        ]  # last state of backward (= first timestep)
        seq_embed = torch.cat(
            (rnn_out_forward, rnn_out_backward), -1
        )  # (B*N*K, rnn_hid_dim*2)
        # unsqueeze
        return seq_embed.view(B, -1)  # (B, N*K, rnn_hid_dim*2)


class CNN(nn.Module):
    def __init__(
        self,
        conv_channels: List[int],
        sentence_length: int,
        embedding_dim: int,
        kernel_sizes: List[int],
        stride=1,
        padding=1,
    ):
        super().__init__()
        layers = [nn.BatchNorm2d(conv_channels[0])]
        for i in range(len(conv_channels) - 1):
            layers.append(
                nn.Conv2d(
                    conv_channels[i],
                    conv_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.BatchNorm2d(conv_channels[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=1))
        layers.pop()
        layers.append(
            nn.AdaptiveMaxPool2d(output_size=(sentence_length, embedding_dim))
        )

        layers.append(nn.Flatten())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, ind) -> torch.Tensor:
        return self.model(x.unsqueeze(2))


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


# class MisinformationModel(nn.Module):
#     def __init__(self, embedding: nn.Module, classifier: nn.Module):
#         super().__init__()
#         self.model = nn.Sequential(embedding, classifier)
#         self._embed = embedding

#     def forward(self, x: torch.Tensor):
#         return self.model(x)

# def evaluate(self, x: torch.Tensor):
#     return self.model(x)
