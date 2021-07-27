from metaphor.adversary.attacks import KSynonymAttack


def main():
    PATH = "/content/drive/My Drive/ucl-msc/dissertation/attacks/"
    val_sentences = []
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
