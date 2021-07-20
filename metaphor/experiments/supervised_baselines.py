import torch.nn as nn
from pathlib import Path
from metaphor.models.common import (
    Bert,
    RNN,
    CNN,
    Glove,
    MLP,
    MeanPooler,
    CustomBertTokenizer,
    StandardTokenizer,
    MisinformationModel,
)
from metaphor.utils.trainer import ClassifierTrainer
from metaphor.data.loading.data_loader import Pheme
import os
import torch

# run all combinations of models
def main():
    tokenizers = [CustomBertTokenizer, StandardTokenizer]
    embeddings = [Bert, Glove]
    models = [MeanPooler, CNN, RNN]
    layers = [
        [[768, 25, 5, 3], [256, 25, 5, 3], [256, 25, 5, 3]],
        [[200, 25, 5, 3], [128, 25, 5, 3], [128, 25, 5, 3]],
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer_args = {
        "lr": 0.001,
        "patience": 10,
        "weight_decay": 0.01,
        "num_epochs": 200,
        "device": device,
        "loss": nn.CrossEntropyLoss(),
    }
    print(Path(__file__).absolute())
    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    print(pheme_path)
    for i in range(2):
        tokenizer = tokenizers[i]()
        data = Pheme(
            file_path=pheme_path, tokenizer=tokenizer, embedder=embeddings[i](tokenizer)
        )
        pool_args = {"tokenizer": tokenizer}
        cnn_args = {
            "conv_channels": [1, 1, 1],
            "sentence_length": tokenizer.max_length,
            "embedding_dim": 256,
            "kernel_sizes": [3, 3],
        }
        glove_cnn_args = {
            "conv_channels": [1, 1, 1],
            "sentence_length": tokenizer.max_length,
            "embedding_dim": 128,
            "kernel_sizes": [3, 3],
        }
        bert_rnn_args = {
            "tokenizer": tokenizer,
            "hidden_dim": 256,
            "embedding_size": 768,
        }
        glove_rnn_args = {
            "tokenizer": tokenizer,
            "hidden_dim": 128,
            "embedding_size": 200,
        }
        args = [
            [pool_args, cnn_args, bert_rnn_args],
            [pool_args, glove_cnn_args, glove_rnn_args],
        ]

        for j in range(3):
            # classifier = nn.Sequential(models[j](**args[j]), MLP(layers))
            classifier = MisinformationModel(models[j](**args[i][j]), MLP(layers[i][j]))
            classifier.to(device)
            print(classifier)
            trainer = ClassifierTrainer(**trainer_args)
            results = trainer.fit(classifier, data.train, data.val)
            print(results)


if __name__ == "__main__":
    main()
