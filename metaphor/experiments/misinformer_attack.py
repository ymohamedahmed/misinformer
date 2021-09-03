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
    model.load_state_dict(torch.load(config.PATH + "bert-mean.npy"))
    embedding.to(device)
    test_sentences = [pheme.data["text"].values[i] for i in pheme.test_indxs]
    # mis.attack(
    #     model=model,
    #     test_labels=pheme.labels[pheme.test_indxs],
    #     test_sentences=test_sentences,
    #     embedding=embedding,
    #     tokenizer=tokenizer,
    # )
    sur_model = MisinformationModel(MaxPooler(**args), MLP(layers))
    sur_model.load_state_dict(torch.load(config.PATH + "bert-max.npy"))
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
