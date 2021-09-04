from metaphor.utils.utils import load_obj, predict
from metaphor.adversary.attacks import Misinformer
from metaphor.models.common import (
    Bert,
    RNN,
    CNN,
    Glove,
    MLP,
    MeanPooler,
    MaxPooler,
    CustomBertTokenizer,
    LogisticRegressor,
    StandardTokenizer,
    MisinformationModel,
    ExpertMixture,
)
from metaphor.data.loading.data_loader import Pheme
from pathlib import Path
import os
import torch
from metaphor.data.loading.data_loader import Pheme, MisinformationPheme
import config

# TODO: perform on each model
# Repeat for different perturbation control in the fixed attack case
# Perform genetic adversarial attack and record the number of attacks before change
def _best_models():
    # Give in the form path, tokenizer, embedding, Aggregator, seed
    paths = [
        "seed-0-bert-rnn-logreg.npy",
        "seed-0-bert-mean-logreg.npy",
        "seed-0-bert-mean-mlp.npy",
    ]
    bert_rnn_args = {
        "hidden_dim": 256,
        "embedding_size": 768,
    }
    args = [bert_rnn_args, {}, {}]
    aggregators = [RNN, MeanPooler, MeanPooler]
    classifiers = [
        LogisticRegressor(768),
        LogisticRegressor(768),
        MLP([768, 25, 5, 3]),
    ]
    models = []
    for i in range(3):
        tokenizer = CustomBertTokenizer()
        embedding = Bert(tokenizer=tokenizer)
        args[i]["tokenizer"] = tokenizer

        model = MisinformationModel(aggregators[i](**args[i]), classifiers[i])
        model.load_state_dict(torch.load(config.PATH + paths[i]))
        models.append([(model, tokenizer, embedding)])
    return models


def fixed_adversary_experiments(pheme, lime_scores):
    file_path = config.PRED_PATH + "fixed_adversary.csv"
    # for each model
    columns = [
        "model",
        "number of concats",
        "max levenhstein",
        "new accuracy",
        "minimum possible accuracy",
        "hit rate",
    ]

    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    pheme = MisinformationPheme(
        file_path=pheme_path,
        tokenizer=lambda x: x,
        embedder=lambda x: torch.zeros((len(x), 200)),
        seed=0,
    )
    test_sentences = [pheme.data["text"].values[i] for i in pheme.test_indxs]
    models = _best_models()
    for (model, tokenizer, embedding) in models:
        for paraphrase in [False, True]:
            for number_of_concats in range(4):
                mis = Misinformer(
                    lime_scores,
                    attacks=[
                        paraphrase,
                        True,
                        number_of_concats > 0,
                    ],
                    number_of_conats=number_of_concats,
                )
                results = mis.attack(
                    model=model,
                    test_labels=pheme.labels[pheme.test_indxs],
                    test_sentences=test_sentences,
                    embedding=embedding,
                    tokenizer=tokenizer,
                )


def genetic_adversary_experiments():
    pass


def adversarial_training_experiments():
    pass


def main():
    train_lime_scores = load_obj(
        "/content/drive/My Drive/ucl-msc/dissertation/checkpoints/train_lime_scores"
    )
    mis = Misinformer(train_lime_scores)
    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    tokenizer = CustomBertTokenizer()
    embedding = Bert(tokenizer=tokenizer)
    layers = [768, 25, 5, 3]
    args = {}
    args["tokenizer"] = tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pheme = MisinformationPheme(
        file_path=pheme_path,
        tokenizer=lambda x: x,
        embedder=lambda x: torch.zeros((len(x), 200)),
    )
    model = MisinformationModel(MeanPooler(**args), MLP(layers))
    model.load_state_dict(torch.load(config.PATH + "seed-0-bert-cnn-nlp.npy"))
    embedding.to(device)
    test_sentences = [pheme.data["text"].values[i] for i in pheme.test_indxs]
    sur_model = MisinformationModel(MaxPooler(**args), MLP(layers))
    sur_model.load_state_dict(torch.load(config.PATH + "seed-0-bert-cnn.npy"))
    sur_tok = tokenizer
    sur_emb = embedding

    mis.genetic_attack(
        model=model,
        surrogate_model=sur_model,
        test_sentences=test_sentences,
        test_labels=pheme.labels[pheme.test_indxs],
        tokenizer=tokenizer,
        embedding=embedding,
        surrogate_tokenizer=sur_tok,
        surrogate_embedding=sur_emb,
    )


if __name__ == "__main__":
    main()
