from metaphor.utils.utils import load_obj, predict
from metaphor.adversary.attacks import Misinformer
from metaphor.utils.trainer import ClassifierTrainer
from tqdm import tqdm
import time
import csv
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
import supervised_baselines
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
        LogisticRegressor(256),
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
        models.append((model, tokenizer, embedding, paths[i]))
    return models


def _surrogate_models(index):
    # from best to worst of the surrogate models
    paths = [
        "seed-0-bert-cnn-logreg.npy",
        "seed-0-bert-cnn-mlp.npy",
    ]
    aggregators = [CNN, CNN]
    classifiers = [
        LogisticRegressor(20 * 210),
        MLP([20 * 210, 25, 5, 3]),
    ]
    args = {
        "conv_channels": [768, 20],
        "output_dim": 210,
        "kernel_sizes": [5],
    }
    tokenizer = CustomBertTokenizer()
    embedding = Bert(tokenizer=tokenizer)
    args["tokenizer"] = tokenizer

    model = MisinformationModel(aggregators[index](**args), classifiers[index])
    model.load_state_dict(torch.load(config.PATH + paths[index]))
    return (model, tokenizer, embedding, paths[index])


def fixed_adversary_experiments(pheme, lime_scores):
    columns = [
        "model",
        "paraphrased",
        "number of concats",
        "max levenhstein",
        "new accuracy",
        "minimum possible accuracy",
        "hit rate",
    ]
    data = [columns]
    test_sentences = [pheme.data["text"].values[i] for i in pheme.test_indxs]
    models = _best_models()
    for (model, tokenizer, embedding, path) in tqdm(models):
        for paraphrase in [False, True]:
            for number_of_concats in range(4):
                mis = Misinformer(
                    lime_scores.copy(),
                    attacks=[
                        paraphrase,
                        True,
                        number_of_concats > 0,
                    ],
                    number_of_concats=number_of_concats,
                )
                results = mis.attack(
                    model=model,
                    test_labels=pheme.labels[pheme.test_indxs],
                    test_sentences=test_sentences,
                    embedding=embedding,
                    tokenizer=tokenizer,
                )
                row = [
                    path,
                    paraphrase,
                    number_of_concats,
                    2,
                    results["new_acc"],
                    results["min_acc"],
                    results["hit_rate"],
                ]
                data.append(row)
    return data


def genetic_adversary_experiments(pheme, lime_scores):
    # different parameters attacking the best model with the worst as the surrogate
    bms = _best_models()
    model, tokenizer, embedding, path = bms[0]
    # surrogate_model, sur_tok, sur_emb, sur_path = _surrogate_models(0)
    surrogate_model, sur_tok, sur_emb, sur_path = bms[1]
    columns = [
        "model",
        "surrogate model",
        "paraphrased",
        "number of concats",
        "max levenhstein",
        "new accuracy",
        "minimum possible accuracy",
        "hit rate",
        "evals per sentence",
        "preds",
    ]
    data = [columns]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_sentences = [pheme.data["text"].values[i] for i in pheme.test_indxs]
    for paraphrase in [False]:
        for number_of_concats in range(4):
            for max_lev in range(1, 4):
                mis = Misinformer(
                    lime_scores.copy(),
                    attacks=[
                        paraphrase,
                        True,
                        number_of_concats > 0,
                    ],
                    max_levenshtein=max_lev,
                    number_of_concats=number_of_concats,
                )
                sur_emb.to(device)
                results = mis.genetic_attack(
                    model=model,
                    surrogate_model=surrogate_model,
                    test_sentences=test_sentences,
                    test_labels=pheme.labels[pheme.test_indxs],
                    tokenizer=tokenizer,
                    embedding=embedding,
                    surrogate_tokenizer=sur_tok,
                    surrogate_embedding=sur_emb,
                    max_generations=30,
                )
                row = [
                    path,
                    sur_path,
                    paraphrase,
                    number_of_concats,
                    max_lev,
                    results["new_acc"],
                    results["min_acc"],
                    results["hit_rate"],
                    results["evals_per_sentence"],
                    results["model_preds"],
                ]
                data.append(row)
    return data


def adversarial_training_experiments(lime_scores, pheme_path):
    # using the best model train with augmentation prob. p to mix in results from _gen_attacks
    # augment tensor is size:  num_sentences x 32 x sentence_length x embedding_dim
    paraphrase, number_of_concats, max_lev = False, 4, 2
    bms = _best_models()
    timestamp = {time.strftime("%d-%m-%y-%H:%M:%S", time.localtime())}
    columns = [
        "model",
        "train acc",
        "unattacked test acc",
        "attacked test acc",
        "hit rate",
        "minimum accuracy",
        "paraphrased",
        "number of concats",
        "max levenshtein",
        "evals per sentence",
        "preds",
    ]
    data = [columns]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 0
    surrogate_model, sur_tok, sur_emb, sur_path = bms[1]
    pheme = MisinformationPheme(
        file_path=pheme_path,
        tokenizer=lambda x: x,
        embedder=lambda x: torch.zeros((len(x), 200)),
    )
    train_sentences = [pheme.data["text"].values[i] for i in pheme.train_indxs]
    test_sentences = [pheme.data["text"].values[i] for i in pheme.test_indxs]
    mis = Misinformer(
        lime_scores.copy(),
        attacks=[
            paraphrase,
            True,
            number_of_concats > 0,
        ],
        max_levenshtein=max_lev,
        number_of_concats=number_of_concats,
    )
    predictions = torch.zeros(
        (
            4,  # number of aggregators
            2,  # number of classifiers
            len(test_sentences),
        )
    )
    baselines_and_labels = torch.zeros((2, len(test_sentences)))

    for emb_ind in range(1):
        for agg_ind in range(4):
            for class_ind in range(2):
                model, tokenizer, embedding = supervised_baselines.instantiate_model(
                    emb_ind, agg_ind, class_ind
                )
                model_name = supervised_baselines.model_name(
                    seed, emb_ind, agg_ind, class_ind
                )

                adv = [y for x in train_sentences for y in mis._gen_attacks(x)[0]]
                tokenized = tokenizer(adv)
                embedding.to(device)
                tokenized = tokenized.to(device)
                adv_emb = embedding(tokenized)
                adv_emb = adv_emb.reshape(
                    (len(train_sentences), 32, tokenized.shape[1], -1)
                )
                data = MisinformationPheme(
                    file_path=pheme_path,
                    tokenizer=tokenizer,
                    embedder=embedding,
                    seed=seed,
                    augmentation=adv_emb,
                )
                trainer = ClassifierTrainer(**supervised_baselines.trainer_args)
                results = trainer.fit(model, data.train, data.val)

                # preds on the unattacked test set
                preds = []
                for x, y in data.test:
                    ind = x[0].to(device)
                    emb = x[1].to(device)
                    y_prime = model(emb, ind).argmax(dim=1).detach().cpu()
                    preds = preds + y_prime.tolist()
                predictions[agg_ind][class_ind] = torch.tensor(preds)

                test_acc = (
                    torch.tensor(preds)
                    .eq(torch.from_numpy(data.labels[data.test_indxs]))
                    .float()
                    .mean()
                    .item()
                )

                # attack
                gen_results = mis.genetic_attack(
                    model=model,
                    surrogate_model=surrogate_model,
                    test_sentences=test_sentences,
                    test_labels=pheme.labels[pheme.test_indxs],
                    tokenizer=tokenizer,
                    embedding=embedding,
                    surrogate_tokenizer=sur_tok,
                    surrogate_embedding=sur_emb,
                    max_generations=30,
                )
                row = [
                    model_name,
                    max(results["train_accuracy"]),
                    test_acc,
                    gen_results["new_acc"],
                    gen_results["hit_rate"],
                    gen_results["min_acc"],
                    paraphrase,
                    number_of_concats,
                    max_lev,
                    gen_results["eval_per_sentence"],
                    gen_results["model_preds"],
                ]
                data.append(row)

    # save preds and labels
    torch.save(
        predictions,
        config.PRED_PATH + f"adversarial_training_test_preds_{timestamp}.npy",
    )
    torch.save(
        baselines_and_labels,
        config.PRED_PATH
        + f"adversarial_training_test_baselines_and_labels_{timestamp}.npy",
    )
    return data


def write_csv(data, file_name):
    f = open(file_name, "w")
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
    f.close()


def main():
    train_lime_scores = load_obj(
        "/content/drive/My Drive/ucl-msc/dissertation/checkpoints/train_lime_scores"
    )
    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    pheme = MisinformationPheme(
        file_path=pheme_path,
        tokenizer=lambda x: x,
        embedder=lambda x: torch.zeros((len(x), 200)),
    )
    timestamp = {time.strftime("%d-%m-%y-%H:%M:%S", time.localtime())}
    # data = fixed_adversary_experiments(pheme, train_lime_scores)
    # write_csv(data, config.PRED_PATH + "fixed_adversary.csv")
    # data = genetic_adversary_experiments(pheme, train_lime_scores)
    # write_csv(data, config.PRED_PATH + f"genetic_adversary_{timestamp}.csv")

    data = adversarial_training_experiments(train_lime_scores, pheme_path)
    write_csv(data, config.PRED_PATH + f"adversarial_training_{timestamp}.csv")

    """
    tokenizer = CustomBertTokenizer()
    embedding = Bert(tokenizer=tokenizer)
    layers = [768, 25, 5, 3]
    args = {}
    args["tokenizer"] = tokenizer
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
    """


if __name__ == "__main__":
    main()
