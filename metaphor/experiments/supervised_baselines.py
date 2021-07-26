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
import wandb

# run all combinations of models
def main():

    tokenizers = [CustomBertTokenizer, StandardTokenizer]
    embeddings = [Bert, Glove]
    models = [MeanPooler, CNN, RNN]
    layers = [
        [[768, 25, 5, 3], [20 * 210, 25, 5, 3], [256, 25, 5, 3]],
        [[200, 25, 5, 3], [20 * 150, 25, 5, 3], [200, 25, 5, 3]],
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer_args = {
        "lr": 0.001,
        "patience": 30,
        "weight_decay": 0.01,
        "num_epochs": 400,
        "device": device,
        "loss": nn.CrossEntropyLoss(),
    }
    PATH = "/content/drive/My Drive/ucl-msc/dissertation/checkpoints/"
    file_names = [
        ["bert-mean.npy", "bert-cnn.npy", "bert-rnn.npy"],
        ["glove-mean.npy", "glove-cnn.npy", "glove-rnn.npy"],
    ]

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
            "conv_channels": [768, 20],
            "sentence_length": tokenizer.max_length,
            "output_dim": 210,
            "kernel_sizes": [5],
            "device": device,
            "tokenizer": tokenizer,
        }
        glove_cnn_args = {
            "conv_channels": [200, 20],
            "sentence_length": tokenizer.max_length,
            "output_dim": 150,
            "kernel_sizes": [7],
            "device": device,
            "tokenizer": tokenizer,
        }
        bert_rnn_args = {
            "tokenizer": tokenizer,
            "hidden_dim": 256,
            "embedding_size": 768,
        }
        glove_rnn_args = {
            "tokenizer": tokenizer,
            "hidden_dim": 200,
            "embedding_size": 200,
        }
        args = [
            [pool_args, cnn_args, bert_rnn_args],
            [pool_args, glove_cnn_args, glove_rnn_args],
        ]

        for j in range(3):
            wandb.init(project="metaphor", entity="youmed")
            config = wandb.config
            config.args = args[i][j]
            config.layers = layers[i][j]
            classifier = MisinformationModel(models[j](**args[i][j]), MLP(layers[i][j]))
            wandb.watch(classifier)
            classifier.to(device)
            print(classifier)
            trainer = ClassifierTrainer(**trainer_args)
            results = trainer.fit(classifier, data.train, data.val)
            print(results)
            print(
                f"max train acc: {max(results['train_accuracy'])}, val acc: {max(results['validation_accuracy'])}"
            )

            # log results and save model
            torch.save(classifier.state_dict(), PATH + file_names[i][j])


if __name__ == "__main__":
    main()
