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
from metaphor.adversary.attacks import KSynonymAttack
import metaphor.experiments.supervised_baselines
import os
import torch
import wandb

# load each checkpoint of the supervised models
def main():
    CHECKPOINT_PATH = "/content/drive/My Drive/ucl-msc/dissertation/checkpoints/"
    ATTACK_PATH = "/content/drive/My Drive/ucl-msc/dissertation/attacks/"
    file_names = [
        ["bert-mean.npy", "bert-cnn.npy", "bert-rnn.npy"],
        ["glove-mean.npy", "glove-cnn.npy", "glove-rnn.npy"],
    ]

    # evaluate model on the 'attacked' validation set
    args = metaphor.experiments.supervised_baselines.args
    layers = metaphor.experiments.supervised_baselines.layers
    models = metaphor.experiments.supervised_baselines.models
    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    pheme = Pheme(
        file_path=pheme_path,
        tokenizer=lambda x: x,
        embedder=lambda x: torch.zeros((len(x), 200)),
    )
    val_sentences = [pheme.data["text"].values[i] for i in pheme.val_indxs]
    labels = torch.tensor(
        [pheme.data["veracity"].values[i] for i in pheme.val_indxs]
    )  # shape: len(val_sentences)
    embeddings = [Bert, Glove]
    tokenizers = [CustomBertTokenizer, StandardTokenizer]
    ATTEMPTS = 5
    preds = torch.zeros((7, 6, ATTEMPTS, len(val_sentences)))
    for k in range(7):
        syn = KSynonymAttack(
            k=k,
            precomputed_sentence_path=ATTACK_PATH + f"synonym_attack_{k}.txt",
            sentences=val_sentences,
        )
        for i in range(2):
            tokenizer = tokenizers[i]()
            embedding = embeddings[i](tokenizer=tokenizer)
            for j in range(3):
                args[i][j]["tokenizer"] = tokenizer
                model = MisinformationModel(models[j](**args[i][j]), MLP(layers[i][j]))
                model.load_state_dict(torch.load(CHECKPOINT_PATH + file_names[i][j]))
                print("Loaded model")
                print("Constructed attack")
                print(val_sentences)

                predictions = metaphor.utils.utils.predict(
                    syn.attacked_sentences,
                    model,
                    tokenizer,
                    embedding,
                )  # shape: attempts x len(val_sentences)
                accuracy = (1.0 * torch.eq(labels, predictions)).min(dim=0)[0].mean()
                print(f"new accuracy: {accuracy}")
                preds[k][(i * 3) + j] = predictions.reshape(
                    (ATTEMPTS, len(val_sentences))
                )


if __name__ == "__main__":
    main()
