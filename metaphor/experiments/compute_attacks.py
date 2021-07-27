from metaphor.adversary.attacks import KSynonymAttack
from pathlib import Path
import os
from metaphor.data.loading.data_loader import Pheme
import torch


def main():
    PATH = "/content/drive/My Drive/ucl-msc/dissertation/attacks/"
    pheme_path = os.path.join(
        Path(__file__).absolute().parent.parent.parent, "data/pheme/processed-pheme.csv"
    )
    pheme = Pheme(
        file_path=pheme_path,
        tokenizer=lambda x: x,
        embedder=lambda x: torch.zeros((len(x), 200)),
    )
    val_sentences = [pheme.data["text"].values[i] for i in pheme.val_indxs]
    for k in range(7):
        asen = KSynonymAttack(
            k=k,
            sentences=val_sentences,
        ).attacked_sentences
        with open(PATH + f"synonym_attack_{k}.txt", "w") as f:
            for sen in asen:
                f.write("%s\n" % sen)


if __name__ == "__main__":
    main()
