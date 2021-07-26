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

# load each checkpoint of the supervised models
def main():
    PATH = "/content/drive/My Drive/ucl-msc/dissertation/checkpoints/"
    file_names = [
        ["bert-mean.npy", "bert-cnn.npy", "bert-rnn.npy"],
        ["glove-mean.npy", "glove-cnn.npy", "glove-rnn.npy"],
    ]
    models = []
    # load checkpoint

    # evaluate model on the validation set

    # evaluate model on the 'attacked' validation set
    model.load_state_dict(torch.load(...))
