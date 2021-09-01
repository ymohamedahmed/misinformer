from lime import lime_text
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
from metaphor.data.loading.data_loader import Pheme, MisinformationPheme
import torch
import os
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
import config
import pickle
from metaphor.utils.utils import forward
import numpy as np


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = CustomBertTokenizer()
embedding = Bert(tokenizer=tokenizer)
pheme_path = "/content/meta-misinformation-detection/data/pheme/processed-pheme.csv"
layers = [768, 25, 5, 3]
args = {}
args["tokenizer"] = tokenizer
pheme = MisinformationPheme(
    file_path=pheme_path,
    tokenizer=lambda x: x,
    embedder=lambda x: torch.zeros((len(x), 200)),
)
model = MisinformationModel(MeanPooler(**args), MLP(layers))
model.load_state_dict(torch.load(config.PATH + "bert-mean.npy"))
embedding.to(device)


def proba(x):
    predictions = forward(
        x,
        model,
        tokenizer,
        embedding,
    )
    return predictions.detach().numpy()


def main():
    class_names = ["false", "unverified", "true"]
    explainer = LimeTextExplainer(class_names=class_names)

    # files: {train,val,test}_{false, unverified, true}_based.csv
    train_sentences = [pheme.data["text"].values[i] for i in pheme.train_indxs]
    global_explainers = {0: {}, 1: {}, 2: {}}
    for sample in tqdm(train_sentences):
        exp = explainer.explain_instance(
            sample, proba, num_features=6, labels=[0, 2], top_labels=3
        )
        for i in range(3):
            for (word, importance) in exp.as_list(i):
                class_global_exp = global_explainers[i]
                if word in class_global_exp:
                    class_global_exp[word].append(importance)
                else:
                    class_global_exp[word] = [importance]

    save_obj(global_explainers, config.PATH + "full_lime_explainer")
    lime_scores_per_class = (
        {}
    )  # a map from word to [false lime score, unverified lime score, true lime score]
    for label in range(3):
        for word in global_explainers[label].keys():
            scores = np.array(global_explainers[label][word])
            lime_score = np.sqrt(np.maximum(0, np.sum(scores)))
            #         lime_score = np.sqrt(np.sum(np.abs(scores)))
            if not (word in lime_scores_per_class):
                lime_scores_per_class[word] = np.zeros((3))
            lime_scores_per_class[word][label] = lime_score

    print(lime_scores_per_class)
    N = len(lime_scores_per_class.keys())
    p_cj = np.zeros((N, 3))
    lime_scores = np.zeros((N, 3))
    for i, word in enumerate(lime_scores_per_class.keys()):
        p_cj[i] = lime_scores_per_class[word] / lime_scores_per_class[word].sum()
        print(lime_scores_per_class[word])
        lime_scores[i, :] = lime_scores_per_class[word]

    # compute entropy
    print(p_cj)
    H = (-p_cj * np.log(p_cj + (10 ** -8))).sum(axis=1)
    coeff = 1 - (H - np.min(H)) / (np.max(H) - np.min(H))
    coeff = coeff.reshape((coeff.shape[0], 1))
    lime_scores = coeff * lime_scores

    for i, word in enumerate(lime_scores_per_class.keys()):
        lime_scores_per_class[word] = lime_scores[i]

    save_obj(lime_scores_per_class, config.PATH + "train_lime_scores")


if __name__ == "__main__":
    main()
