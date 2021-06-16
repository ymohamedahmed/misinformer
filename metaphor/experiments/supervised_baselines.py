import torch.nn as nn
from ..models.common import Bert, RNN, CNN, Glove, MLP, MeanPooler
from ..models.tokenizers import BertTokenizer, Tokenizer

# run all combinations of models
tokenizers = [BertTokenizer, Tokenizer]
embedding = [Bert, Glove]
models = [MeanPooler, RNN, CNN]
for i in range(2):
    for j in range(3):
        classifier = nn.Sequential()
        model = nn.Sequential([embedding[i](), models[j]()])
        # fit model
