import torch.nn as nn
from ..models.common import Bert, RNN, CNN, Glove, MLP, MeanPooler
from ..models.tokenizers import BertTokenizer, Tokenizer
from metaphor.utils.trainer import Trainer
from metaphor.data.loading.data_loader import Pheme

# run all combinations of models
tokenizers = [BertTokenizer, Tokenizer]
embedding = [Bert, Glove]
models = [MeanPooler, RNN, CNN]
layers = [25, 5, 3]
trainer_args = {}
pheme_path = ""
for i in range(2):
    tokenizer = tokenizers[i]()
    data = Pheme(
        file_path=pheme_path, tokenizer=tokenizer, embedder=embedding[i](tokenizer)
    )
    for j in range(3):
        classifier = nn.Sequential(models[j](), MLP(layers))
        trainer = Trainer(**trainer_args)
        trainer.fit(classifier, data.train, data.val)
