import torch
import torch.nn as nn
import numpy as np
from typing import List
from metaphor.models.common.tokenize import Tokenizer
from metaphor.utils.utils import load_obj
from enum import Enum
import gensim.downloader as api
from gensim.utils import tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import string
import metaphor


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class Attack:
    pass


class MisinformerMode(Enum):
    UNTARGETED = 1
    FALSE = 2
    UNVERIFIED = 3
    TRUE = 4


class ConcatenationAttack(Attack):
    def __init__(self, lime_scores, number_of_concats: int):
        self.lime_scores = lime_scores
        self.number_of_concats = number_of_concats

    def attack(self, sentences):
        attacked_sentences = []
        for x in sentences:
            # sample concatenation strings
            indxs = np.random.choice(
                len(self.lime_scores.keys()),
                size=self.number_of_concats,
                p=softmax(self.lime_scores.values),
            )
            attack = list(self.lime_scores.keys())[indxs]
            attacked_sentences.append(x + " " + attack)
        return attacked_sentences


class CharAttack(Attack):
    def __init__(self, lime_scores, max_levenshtein=2, max_number_of_words=3):
        self.lime_scores = lime_scores
        self.max_lev = max_levenshtein
        self.max_number_of_words = max_number_of_words

    def _attack(self, word):
        lower_chars = list(string.ascii_lowercase)
        for _ in range(self.max_lev):
            # insertion, deletion or sub
            ind = np.random.randint(low=0, high=len(word))
            p = np.random.uniform()
            if p < 1 / 3:
                # insertion
                word = word[:ind] + np.random.choice(lower_chars) + word[ind:]
            elif p < 2 / 3:
                # deletion
                word = word[:ind] + word[ind + 1 :]
            else:
                # sub
                word = word[:ind] + np.random.choice(lower_chars) + word[ind + 1 :]
        return word

    def attack(self, sentences: List[str]):
        scores = []
        attacked = []
        for sentence in sentences:
            for word in sentence.split():
                score = self.lime_scores[word] if word in self.lime_scores else 0
                scores.append(score)
            scores = np.array(scores)
            attack_words_indxs = np.random.choice(
                a=len(scores), size=self.max_number_of_words, p=softmax(scores)
            )
            attacked_sentence = sentence.split()
            for i in attack_words_indxs:
                attacked_sentence[i] = self._attack(attacked_sentence[i])

            attacked.append(" ".join(attacked_sentence))
        return attacked


class ParaphraseAttack(Attack):
    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        attempts: int = 16,
        path: str = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.path = path
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.device = device
        self.model.to(device)
        self.attempts = attempts

    def attack(self, sentences: List[str]):
        if self.path is not None:
            with open(self.path) as f:
                lines = f.readlines()
                return [x.strip() for x in lines]
        attacked_sentences = []
        for sentence in sentences:
            encoding = self.tokenizer.encode_plus(
                sentence, pad_to_max_length=True, return_tensors="pt"
            )
            input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding[
                "attention_mask"
            ].to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=256,
                do_sample=True,
                top_k=200,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=self.attempts,
            )

            lines = []
            for output in outputs:
                line = self.tokenizer.decode(
                    output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                attacked_sentences.append(line)

        return attacked_sentences


class Misinformer(Attack):
    def __init__(
        self,
        lime_scores,
        target=MisinformerMode.TRUE,
        attacks=[
            True,
            True,
            True,
        ],  # whether or not to use paraphrase attack, char attack and/or concat attack
        number_of_concats=4,
    ):
        # lime scores is of the form word -> [false-score, unverified-score, true-score]
        self.lime_scores = lime_scores
        indices = {
            MisinformerMode.FALSE: 0,
            MisinformerMode.UNVERIFIED: 1,
            MisinformerMode.TRUE: 2,
        }
        agg = (
            lambda array: np.mean(array)
            if target == MisinformerMode.UNTARGETED
            else array[indices[target]]
        )
        self.attacks = attacks
        for k in self.lime_scores.keys():
            self.lime_scores[k] = agg(self.lime_scores[k])
        self.paraphraser = ParaphraseAttack()
        self.char_attack = CharAttack(self.lime_scores)
        self.concat_attack = ConcatenationAttack(
            self.lime_scores, number_of_concats=number_of_concats
        )

    def attack(self, model, surrogate_model, test_set):
        # test_set should be the set of strings and labels
        scores = []
        for x, y in test_set:
            for word in x.split():
                score = self.lime_scores[word] if word in self.lime_scores else 10 ** -5
                scores.append(score)

            if self.attacks[0]:
                attacked = self.paraphraser.attack([x]) + [x for _ in range(16)]
            else:
                attacked = [x for _ in range(32)]
            # a batch of 32 strings, 16 have been paraphrased, 16 are the original
            if self.attacks[1]:
                attacked = self.char_attack.attack(attacked)
            if self.attacks[2]:
                attacked = self.concat_attack.attack(attacked)
            print(attacked)


# Not in use
class KSynonymAttack(Attack):
    def __init__(
        self,
        k: int,
        attempts: int = 5,
        N: int = 3,
        batch_size: int = 128,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        path: str = None,
    ):
        """
        k: the number of words to substitute
        embedding: either Glove or BERT
        model: trained model
        max_length: length of the longest sentence
        attempts: maximum number of attacks (i.e. number of different substitutions attempted)
        N: number of nearest neighbours to consider when picking synonym in embedding space
        e.g. KSynonymAttack(k=5, embedding=Glove, tokenizer=StandardTokenizer, model=...,)
        """
        np.random.seed(0)
        self.k = k
        self.attempts = attempts
        self.N = N
        self.synonym_model = api.load("glove-wiki-gigaword-300")
        self.device = device
        self.path = path

    def attack(self, sentences: List[str]):
        if self.path is not None:
            with open(self.path) as f:
                lines = f.readlines()
                return [x.strip() for x in lines]

        attacked_sentences = []
        self.num_sentences = len(sentences)
        # indices of words to substitute
        for sentence in tqdm(sentences):
            sentence = [x for x in tokenize(sentence.lower())]
            for i in range(self.attempts):
                indxs = np.random.choice(len(sentence), size=self.k)
                new_sent = sentence.copy()
                for j in indxs:
                    # if the words isn't in the embedding space don't replace it
                    synonyms = [sentence[j]]
                    try:
                        self.synonym_model.get_index(sentence[j])
                        synonyms = [
                            x[0]
                            for x in self.synonym_model.most_similar(sentence[j])[
                                : self.N
                            ]
                        ]
                    except KeyError:
                        pass

                    new_sent[j] = np.random.choice(synonyms)
                attacked_sentences.append(" ".join(new_sent))
        return attacked_sentences


if __name__ == "__main__":
    train_lime_scores = load_obj(
        "/content/drive/My Drive/ucl-msc/dissertation/checkpoints/train_lime_scores"
    )
    mis = Misinformer(train_lime_scores)
    fake_test = [("The quick brown fox jumps", 1), ("Hello world dot com", 2)]
    mis.attack(model=None, surrogate_model=None, test_set=fake_test)
