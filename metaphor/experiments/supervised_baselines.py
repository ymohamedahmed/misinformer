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
    LogisticRegressor,
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
    HardPheme,
)
import scipy.stats
import os
import torch
import wandb
import config

tokenizers = [CustomBertTokenizer, StandardTokenizer, StandardTokenizer]
embeddings = [Bert, Glove, Word2Vec]
aggregators = [MeanPooler, MaxPooler, CNN, RNN]
classifiers = [MLP, LogisticRegressor]
layers = [
    [[768, 25, 5, 3], [768, 25, 5, 3], [20 * 210, 25, 5, 3], [256, 25, 5, 3]],
    [[200, 25, 5, 3], [200, 25, 5, 3], [20 * 150, 25, 5, 3], [200, 25, 5, 3]],
    [[300, 25, 5, 3], [300, 25, 5, 3], [20 * 150, 25, 5, 3], [300, 25, 5, 3]],
]
# TOTAL models: 24

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
    NUM_SEEDS = 5
    predictions = None
    baselines_and_labels = None
    for seed in range(NUM_SEEDS):
        for i in range(3):
            tokenizer = tokenizers[i]()
            data = MisinformationPheme(
                file_path=pheme_path,
                tokenizer=tokenizer,
                embedder=embeddings[i](tokenizer),
                seed=seed,
            )
            if predictions is None:
                test_sentences = [data.data["text"].values[i] for i in data.test_indxs]
                predictions = torch.zeros(
                    (
                        NUM_SEEDS,
                        len(tokenizers),
                        len(aggregators),
                        len(classifiers),
                        len(test_sentences),
                    )
                )
                baselines_and_labels = torch.zeros((NUM_SEEDS, 2, len(test_sentences)))

            for j in range(4):
                for classifier_ind in range(2):
                    wandb.init(project="metaphor", entity="youmed", reinit=True)
                    args[i][j]["tokenizer"] = tokenizer
                    wandb_config = wandb.config
                    wandb_config.args = args[i][j]
                    wandb_config.layers = layers[i][j]
                    classifier_model = (
                        MLP(layers[i][j])
                        if classifier_ind == 0
                        else LogisticRegressor(layers[i][j][0])
                    )
                    classifier = MisinformationModel(
                        aggregators[j](**args[i][j]), classifier_model
                    )
                    wandb.watch(classifier)
                    classifier.to(device)
                    print(classifier)
                    trainer = ClassifierTrainer(**trainer_args)
                    results = trainer.fit(classifier, data.train, data.val)
                    print(results)

                    # log results and save model
                    torch.save(
                        classifier.state_dict(),
                        config.PATH
                        + f"seed_{seed}_"
                        + file_names[i][j]
                        + type(classifier_model).__name__,
                    )
                    preds = []
                    for x, y in data.test:
                        ind = x[0].to(device)
                        emb = x[1].to(device)
                        y_prime = classifier(emb, ind).argmax(dim=1).detach().cpu()
                        preds = preds + y_prime.tolist()
                    predictions[(i * len(tokenizers)) + j] = torch.tensor(preds)
                    test_acc = (
                        torch.tensor(preds)
                        .eq(torch.from_numpy(data.labels[data.test_indxs]))
                        .float()
                        .mean()
                        .item()
                    )
                    print(
                        f"max train acc: {max(results['train_accuracy'])}, val acc: {max(results['validation_accuracy'])}, test acc: {test_acc}"
                    )
        # most common baseline
        pheme = MisinformationPheme(
            file_path=pheme_path,
            tokenizer=lambda x: x,
            embedder=lambda x: torch.zeros((len(x), 200)),
        )
        labels = pheme.labels[pheme.train_indxs]
        baselines_and_labels[seed][0] = torch.from_numpy(
            scipy.stats.mode(labels)[0]
        ) * torch.ones((len(pheme.labels[pheme.test_indxs])))
        baselines_and_labels[seed][1] = torch.from_numpy(pheme.labels[pheme.test_indxs])
        torch.save(predictions, config.PRED_PATH + f"test_predictions.npy")
        torch.save(
            baselines_and_labels, config.PRED_PATH + f"test_baselines_and_label.npy"
        )


#  retrain on the harder task
if __name__ == "__main__":
    main()
