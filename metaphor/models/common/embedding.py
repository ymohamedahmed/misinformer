from metaphor.models.common.pooling import Pooler
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np
import gensim
import gensim.downloader as api
from metaphor.models.common.tokenize import Tokenizer, BertTokenizer
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sentence - Glove/Bert -> Tensor of embeddings - CNN/RNN/MeanPool -> Single vector
# e.g. nn.Sequential([Bert(), RNN()]) or nn.Sequential([WordEmbedding(Glove)])


class Bert(nn.Module):
    def __init__(self, tokenizer: BertTokenizer):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, x: torch.FloatTensor):
        return self.model(x)


class Glove(nn.Module):
    def __init__(self, tokenizer: Tokenizer):
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
    def __init__(self, tokenizer: Tokenizer, hidden_dim: int, embedding_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        # self.dictionary = dictionary
        self.hidden_dim = hidden_dim // 2  # assuming bidirectional
        # self.padding_token = self.dictionary["PAD"]

        # self.text_emb_size = embedding_weights.shape[-1]

        # RNN
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.rnn_hid_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
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
            text_embedding,
            self.tokenizer.sentence_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_out_packed, _ = self.rnn(text_embedding_packed)
        rnn_out, _ = pad_packed_sequence(
            rnn_out_packed, batch_first=True
        )  # (B*N*K, max_seq_len, rnn_hid_dim*2)

        # concat forward and backward results (takes output states)
        seq_len_indices = [length - 1 for length in self.tokenizer.sentence_lengths]
        batch_indices = [i for i in range(B * NK)]
        rnn_out_forward = rnn_out[
            batch_indices, seq_len_indices, : self.rnn_hid_dim
        ]  # last state of forward (not padded)
        rnn_out_backward = rnn_out[
            :, 0, self.rnn_hid_dim :
        ]  # last state of backward (= first timestep)
        seq_embed = torch.cat(
            (rnn_out_forward, rnn_out_backward), -1
        )  # (B*N*K, rnn_hid_dim*2)
        # unsqueeze
        return seq_embed.view(B, NK, -1)  # (B, N*K, rnn_hid_dim*2)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()


class MLP(nn.Module):
    def __init__(self, layers: List[int]):
        super().__init__()


# class MisinformationModel(nn.Module):
#     def __init__(self, embedding: nn.Module, classifier: nn.Module):
#         super().__init__()
#         self.model = nn.Sequential(embedding, classifier)
#         self._embed = embedding

#     def forward(self, x: torch.Tensor):
#         return self.model(x)

# def evaluate(self, x: torch.Tensor):
#     return self.model(x)
