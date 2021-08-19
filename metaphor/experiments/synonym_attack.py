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
    ExpertMixture,
)
from metaphor.utils.trainer import ClassifierTrainer
from metaphor.utils.utils import predict
from metaphor.data.loading.data_loader import Pheme, MisinformationPheme
from metaphor.adversary.attacks import KSynonymAttack, ParaphraseAttack
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
    em_file_names = [
        ["em-bert-mean.npy", "em-bert-cnn.npy", "em-bert-rnn.npy"],
        ["em-glove-mean.npy", "em-glove-cnn.npy", "em-glove-rnn.npy"],
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # evaluate model on the 'attacked' validation set
    args = metaphor.experiments.supervised_baselines.args
    layers = metaphor.experiments.supervised_baselines.layers
    models = metaphor.experiments.supervised_baselines.models
    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    pheme = MisinformationPheme(
        file_path=pheme_path,
        tokenizer=lambda x: x,
        embedder=lambda x: torch.zeros((len(x), 200)),
    )
    val_sentences = [pheme.data["text"].values[i] for i in pheme.val_indxs]
    labels = torch.from_numpy(pheme.labels[pheme.val_indxs])
    # labels = torch.tensor(
    #     [pheme.data["veracity"].values[i] for i in pheme.val_indxs]
    # )  # shape: len(val_sentences)
    embeddings = [Bert, Glove]
    tokenizers = [CustomBertTokenizer, StandardTokenizer]
    ATTEMPTS = 5
    preds = torch.zeros((6, ATTEMPTS, len(val_sentences)))
    syn = ParaphraseAttack(
        path=ATTACK_PATH + "paraphrase_attack.txt",
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

            predictions = predict(
                syn.attacked_sentences,
                model,
                tokenizer,
                embedding,
            )  # shape: attempts x len(val_sentences)
            print(len(syn.attacked_sentences))
            print(predictions.shape)
            print(len(labels))
            predictions = predictions.reshape((len(val_sentences), ATTEMPTS)).T
            accuracy = (1.0 * torch.eq(labels, predictions)).min(dim=0)[0].mean()
            print(f"new accuracy: {accuracy}")
            preds[(i * 3) + j] = predictions

            n_topics = 9
            mlp_layers = layers[i][j].copy()
            mlp_layers.pop()
            mlp_layers.append(9)
            expert_mixture = ExpertMixture(
                aggregator=models[j](**args[i][j]),
                n_topics=n_topics,
                models=[
                    MisinformationModel(models[j](**args[i][j]), MLP(layers[i][j]))
                    for _ in range(n_topics)
                ],
                topic_selector=MisinformationModel(
                    models[j](**args[i][j]), MLP(mlp_layers)
                ),
            )
            expert_mixture.load_state_dict(
                torch.load(CHECKPOINT_PATH + em_file_names[i][j])
            )
            # expert_mixture.to(device)

            predictions = predict(
                syn.attacked_sentences,
                expert_mixture,
                tokenizer,
                embedding,
            )  # shape: attempts x len(val_sentences)
            predictions = predictions.reshape((len(val_sentences), ATTEMPTS)).T
            accuracy = (1.0 * torch.eq(labels, predictions)).min(dim=0)[0].mean()
            print(f"em acc: {accuracy}")
    torch.save(preds, ATTACK_PATH + "paraphrase_attack_preds.npy")


if __name__ == "__main__":
    main()
