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
    Tokenizer,
    MisinformationModel,
)
from metaphor.utils.trainer import ClassifierTrainer
from metaphor.data.loading.data_loader import Pheme
import os
import torch

# run all combinations of models
def main():
    tokenizers = [CustomBertTokenizer, Tokenizer]
    embeddings = [Bert, Glove]
    models = [MeanPooler, RNN, CNN]
    layers = [25, 5, 3]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer_args = {
        "lr": 0.0001,
        "patience": 10,
        "weight_decay": 0.01,
        "num_epochs": 200,
        "device": device,
        "loss": nn.CrossEntropyLoss,
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
            "conv_channels": [4, 4, 4],
            "sentence_length": tokenizer.max_length,
            "embedding_dim": 128,
            "kernel_sizes": [3, 3],
        }
        rnn_args = {"tokenizer": tokenizer, "hidden_dim": 4, "embedding_size": 128}
        args = [pool_args, cnn_args, rnn_args]

        for j in range(3):
            # classifier = nn.Sequential(models[j](**args[j]), MLP(layers))
            classifier = MisinformationModel(models[j](**args[j]), MLP(layers))
            classifier.to(device)
            print(classifier)
            print(classifier.summary())
            trainer = ClassifierTrainer(**trainer_args)
            results = trainer.fit(classifier, data.train, data.val)
            print(results)


if __name__ == "__main__":
    main()
