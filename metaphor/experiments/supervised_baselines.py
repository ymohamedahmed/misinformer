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
from metaphor.data.loading.data_loader import (
    MisinformationPheme,
    TopicPheme,
    PerTopicMisinformation,
)
import os
import torch
import wandb

tokenizers = [CustomBertTokenizer, StandardTokenizer]
embeddings = [Bert, Glove]
models = [MeanPooler, CNN, RNN]
layers = [
    [[768, 25, 5, 3], [20 * 210, 25, 5, 3], [256, 25, 5, 3]],
    [[200, 25, 5, 3], [20 * 150, 25, 5, 3], [200, 25, 5, 3]],
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
bert_rnn_args = {
    "hidden_dim": 256,
    "embedding_size": 768,
}
glove_rnn_args = {
    "hidden_dim": 200,
    "embedding_size": 200,
}
args = [
    [pool_args, cnn_args, bert_rnn_args],
    [pool_args, glove_cnn_args, glove_rnn_args],
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
    PATH = "/content/drive/My Drive/ucl-msc/dissertation/checkpoints/"
    file_names = [
        ["bert-mean.npy", "bert-cnn.npy", "bert-rnn.npy"],
        ["glove-mean.npy", "glove-cnn.npy", "glove-rnn.npy"],
    ]

    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    for i in range(2):
        tokenizer = tokenizers[i]()
        data = MisinformationPheme(
            file_path=pheme_path, tokenizer=tokenizer, embedder=embeddings[i](tokenizer)
        )

        for j in range(3):
            wandb.init(project="metaphor", entity="youmed", reinit=True)
            args[i][j]["tokenizer"] = tokenizer
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
            # print(
            # f"max train acc: {max(results['train_accuracy'])}, val acc: {max(results['validation_accuracy'])}"
            # )

            # log results and save model
            torch.save(classifier.state_dict(), PATH + file_names[i][j])

            # train mixture of experts
            topic_pheme = TopicPheme(
                file_path=pheme_path,
                tokenizer=tokenizer,
                embedder=embeddings[i](tokenizer),
            )

            # change MLP to classify topics
            mlp_layers = layers[i][j].copy()
            mlp_layers.pop()
            mlp_layers.append(9)
            # ptm = PerTopicMisinformation(
            #     file_path=pheme_path,
            #     tokenizer=tokenizer,
            #     embedder=embeddings[i](tokenizer),
            # )
            n_topics = 9
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
            expert_mixture.to(device)
            expert_mixture.fit(trainer, topic_pheme, data.per_topic())
            em_acc, em_loss = trainer._evaluate_validation(expert_mixture, data.val)
            acc, loss = trainer._evaluate_validation(classifier, data.val)
            print(f"em acc: {em_acc}, standard: {acc}")
            # results = trainer.fit(expert_mixture, data.train, data.val)
            # print(f"max train acc: {acc}, val acc: {loss}")

            torch.save(expert_mixture.state_dict(), PATH + "em-" + file_names[i][j])

    # train each of the models with the defenses


if __name__ == "__main__":
    main()
