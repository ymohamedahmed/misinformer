import torch.nn as nn
from pathlib import Path
from metaphor.models.common import (
    Bert,
    RNN,
    CNN,
    Glove,
    Word2Vec,
    MLP,
    MeanPooler,
    MaxPooler,
    CustomBertTokenizer,
    StandardTokenizer,
    MisinformationModel,
    ExpertMixture,
)
from metaphor.utils.trainer import ClassifierTrainer
from metaphor.data.loading.data_loader import (
    MisinformationPheme,
    TopicPheme,
    PerTopicMisinformation,
)
import scipy.stats
import os
import torch
import wandb
import config

tokenizers = [CustomBertTokenizer, StandardTokenizer, StandardTokenizer]
embeddings = [Bert, Glove, Word2Vec]
models = [MeanPooler, MaxPooler, CNN, RNN]
layers = [
    [[768, 25, 5, 3], [768, 25, 5, 3], [20 * 210, 25, 5, 3], [256, 25, 5, 3]],
    [[200, 25, 5, 3], [200, 25, 5, 3], [20 * 150, 25, 5, 3], [200, 25, 5, 3]],
    [[300, 25, 5, 3], [300, 25, 5, 3], [20 * 150, 25, 5, 3], [300, 25, 5, 3]],
]
pool_args = {}
cnn_args = {
    "conv_channels": [768, 20],
    "output_dim": 210,
    "kernel_sizes": [5],
}
glove_cnn_args = {
    "conv_channels": [200, 20],
    "output_dim": 150,
    "kernel_sizes": [7],
}
word2vec_cnn_args = {
    "conv_channels": [300, 20],
    "output_dim": 150,
    "kernel_sizes": [7],
}
bert_rnn_args = {
    "hidden_dim": 256,
    "embedding_size": 768,
}
glove_rnn_args = {
    "hidden_dim": 200,
    "embedding_size": 200,
}
word2vec_rnn_args = {
    "hidden_dim": 300,
    "embedding_size": 300,
}
args = [
    [pool_args, {}, cnn_args, bert_rnn_args],
    [pool_args, {}, glove_cnn_args, glove_rnn_args],
    [pool_args, {}, word2vec_cnn_args, word2vec_rnn_args],
]


# run all combinations of models
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer_args = {
        "lr": 0.001,
        "patience": 30,
        "weight_decay": 0.01,
        "num_epochs": 400,
        "device": device,
        "loss": nn.CrossEntropyLoss(),
    }
    file_names = [
        ["bert-mean.npy", "bert-max.npy", "bert-cnn.npy", "bert-rnn.npy"],
        ["glove-mean.npy", "glove-max.npy", "glove-cnn.npy", "glove-rnn.npy"],
        [
            "word2vec-mean.npy",
            "word2vec-max.npy",
            "word2vec-cnn.npy",
            "word2vec-rnn.npy",
        ],
    ]

    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    predictions = None
    for i in range(3):
        tokenizer = tokenizers[i]()
        data = MisinformationPheme(
            file_path=pheme_path, tokenizer=tokenizer, embedder=embeddings[i](tokenizer)
        )
        if predictions is None:
            test_sentences = [data.data["text"].values[i] for i in data.test_indxs]
            predictions = torch.zeros((14, len(test_sentences)))

        for j in range(4):
            wandb.init(project="metaphor", entity="youmed", reinit=True)
            args[i][j]["tokenizer"] = tokenizer
            wandb_config = wandb.config
            wandb_config.args = args[i][j]
            wandb_config.layers = layers[i][j]
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
            torch.save(
                classifier.state_dict(),
                config.PATH + file_names[i][j],
            )
            preds = []
            for x, y in data.test:
                ind = x[0].to(device)
                emb = x[1].to(device)
                y_prime = classifier(emb, ind).argmax(dim=1).detach().cpu()
                preds = preds + y_prime.tolist()
            predictions[(i * len(tokenizers)) + j] = torch.tensor(preds)

    # most common baseline
    pheme = MisinformationPheme(
        file_path=pheme_path,
        tokenizer=lambda x: x,
        embedder=lambda x: torch.zeros((len(x), 200)),
    )
    labels = pheme.labels[pheme.train_indxs]
    predictions[13] = torch.from_numpy(scipy.stats.mode(labels)[0]) * torch.ones(
        (len(pheme.labels[pheme.test_indxs]))
    )
    predictions[14] = pheme.labels[pheme.test_indxs]
    torch.save(predictions, config.PRED_PATH + "test_predictions.npy")


#  retrain on the harder task
if __name__ == "__main__":
    main()
