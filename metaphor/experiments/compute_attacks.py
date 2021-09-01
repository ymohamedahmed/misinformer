from metaphor.adversary.attacks import (
    KSynonymAttack,
    ParaphraseAttack,
    ConcatenationAttack,
)
from pathlib import Path
import os
from metaphor.data.loading.data_loader import Pheme
import torch

# For each attack compute it on the train, validation and test set
# The train set allows for adversarial training (consider it as a data-augmentation)


def _paraphrase():
    pass


def _concat():
    pass


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
    asen = ParaphraseAttack().attack(val_sentences)
    with open(PATH + "val_paraphrase_attack.txt", "w") as f:
        for sen in asen:
            f.write("%s\n" % sen)


if __name__ == "__main__":
    main()
