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
import metaphor.experiments.supervised_baselines
import os
import torch
import wandb

# load each checkpoint of the supervised models
def main():
    PATH = "/content/drive/My Drive/ucl-msc/dissertation/checkpoints/"
    file_names = [
        ["bert-mean.npy", "bert-cnn.npy", "bert-rnn.npy"],
        ["glove-mean.npy", "glove-cnn.npy", "glove-rnn.npy"],
    ]
    # load checkpoint

    # evaluate model on the validation set

    # evaluate model on the 'attacked' validation set
    args = metaphor.experiments.supervised_baselines.args
    layers = metaphor.experiments.supervised_baselines.layers
    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    pheme = Pheme(
        file_path=pheme_path, tokenizer=lambda x: x, embedder=lambda x: x
    )
    val_sentences = [pheme.data['text'].values[i] for i in pheme.val_indxs]
    labels = [pheme.data['veracity'].values[i] for i in pheme.val_indxs] # shape: len(val_sentences)
    embedding = [Bert, Glove]
    tokenizers = [CustomBertTokenizer, StandardTokenizer]
    for i in range(2):
        tokenizer = tokenizers[i]() 
        embedding = embeddings[i](tokenizer=tokenizer)
        for j in range(3):
            args[i][j]['tokenizer'] = tokenizer
            model = MisinformationModel(models[j](**args[i][j]), MLP(layers[i][j]))
            model.load_state_dict(torch.load(PATH + file_names[i][j]))
            syn = KSynonymAttack(k=5
                embedding=embedding,
                tokenizer=tokenizer,
                model=model)

            predictions = syn.attack(val_sentences) # shape: attempts x len(val_sentences)
            accuracy = (1.*torch.eq(labels,predictions)).min(dim=0)[0].mean()
            print(f"new accuracy: {accuracy}")