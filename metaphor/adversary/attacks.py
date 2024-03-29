import torch
import torch.nn as nn
import numpy as np
from typing import List
from metaphor.models.common.tokenize import Tokenizer
from metaphor.utils.utils import load_obj, predict, forward
from enum import Enum
import gensim.downloader as api
from gensim.utils import tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import string
import metaphor
import metaphor.experiments.config as config
import time
import logging


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


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
        self._p = softmax(np.array(list(self.lime_scores.values())))

    def _choose(self, size):
        indxs = np.random.choice(
            len(self.lime_scores.keys()),
            size=size,
            p=self._p,
        )
        return [list(self.lime_scores.keys())[i] for i in indxs]

    def attack(self, sentences):
        attacked_sentences = []
        masks = []
        for x in sentences:
            # sample concatenation strings
            attack = self._choose(self.number_of_concats)
            mask = np.zeros((len(x.split()) + len(attack)))
            mask[-len(attack) :] = 1
            attack = " ".join(attack)
            attacked_sentences.append(x + " " + attack)
            masks.append(mask)
        return attacked_sentences, masks


class CharAttack(Attack):
    def __init__(self, neg_lime_scores, max_levenshtein=1, max_number_of_words=3):
        self.lime_scores = neg_lime_scores
        self.max_lev = max_levenshtein
        self.max_number_of_words = max_number_of_words

    def _attack(self, orig_word):
        lower_chars = list(string.ascii_lowercase)
        word = orig_word
        for _ in range(min(self.max_lev, len(orig_word.strip()) - 1)):
            # insertion, deletion or sub
            if len(orig_word.strip()) > 0:
                ind = np.random.randint(low=0, high=len(orig_word.strip()))
                p = np.random.uniform()
                if p < 1 / 3:
                    # insertion
                    word = (
                        orig_word[:ind]
                        + np.random.choice(lower_chars)
                        + orig_word[ind:]
                    )
                elif p < 2 / 3:
                    # deletion
                    word = orig_word[:ind] + orig_word[ind + 1 :]
                else:
                    # sub
                    word = (
                        orig_word[:ind]
                        + np.random.choice(lower_chars)
                        + orig_word[ind + 1 :]
                    )
        return word

    def attack(self, sentences: List[str]):
        attacked = []
        masks = []
        for sentence in sentences:
            scores = []
            mask = np.zeros(len(sentence.split()))
            for word in sentence.split():
                score = self.lime_scores[word] if word in self.lime_scores else 0
                scores.append(score)
            scores = np.array(scores)
            attack_words_indxs = np.random.choice(
                a=len(scores), size=self.max_number_of_words, p=softmax(scores)
            )
            mask[attack_words_indxs] = 1
            attacked_sentence = sentence.split()
            for i in list(set(attack_words_indxs)):
                attacked_sentence[i] = self._attack(attacked_sentence[i])

            attacked.append(" ".join(attacked_sentence))
            masks.append(mask)
        return attacked, masks


class ParaphraseAttack(Attack):
    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        attempts: int = 4,
        path: str = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Vamsi/T5_Paraphrase_Paws", use_fast=True
        )
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
                top_k=10,
                top_p=0.99,
                early_stopping=True,
                num_return_sequences=self.attempts,
            )

            for output in outputs:
                line = self.tokenizer.decode(
                    output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                attacked_sentences.append(line)

        return attacked_sentences


class Misinformer(Attack):
    def __init__(
        self,
        pos_lime_scores,
        neg_lime_scores,
        target=MisinformerMode.TRUE,
        attacks=[
            True,
            True,
            True,
        ],  # whether or not to use paraphrase attack, char attack and/or concat attack
        number_of_concats=4,
        max_levenshtein=2,
        max_number_of_words=3,
        seed=0,
        log_to_file=True,
        pre_initialised_paraphraser=None,
    ):
        # lime scores is of the form word -> [false-score, unverified-score, true-score]
        np.random.seed(seed)
        if log_to_file:
            timestamped_file = f'misinformer-log-{time.strftime("%d-%m-%y-%H:%M:%S", time.localtime())}.txt'
            logging.basicConfig(
                filename=config.LOGGING_PATH + timestamped_file, level=logging.DEBUG
            )
            logging.info(
                f"Settings: para-{attacks[0]}-char-{attacks[1]}-concat-{attacks[2]}-number_concats-{number_of_concats}-max_lev-{max_levenshtein}-max_num_words-{max_number_of_words}"
            )

        self.pos_lime_scores = pos_lime_scores
        self.neg_lime_scores = neg_lime_scores
        if target == MisinformerMode.UNTARGETED:
            raise ValueError(
                "Untargeted attacks are no longer supported; this is primarily for a True attack"
            )
        indices = {
            MisinformerMode.FALSE: 0,
            MisinformerMode.UNVERIFIED: 1,
            MisinformerMode.TRUE: 2,
        }
        self.target_label = indices[target]
        agg = (
            lambda array: np.mean(array)
            if target == MisinformerMode.UNTARGETED
            else array[indices[target]]
        )
        self.attacks = attacks

        for s in [self.neg_lime_scores, self.pos_lime_scores]:
            for k in s.keys():
                s[k] = agg(s[k])
        if pre_initialised_paraphraser is not None:
            self.paraphraser = pre_initialised_paraphraser
        else:
            self.paraphraser = ParaphraseAttack()
        self.char_attack = CharAttack(
            self.neg_lime_scores,
            max_levenshtein=max_levenshtein,
            max_number_of_words=max_number_of_words,
        )
        self.concat_attack = ConcatenationAttack(
            self.pos_lime_scores, number_of_concats=number_of_concats
        )

    #  TODO: Add support for changing number of attacked sentences generated
    #  and measure how effectiveness changes

    def _gen_attacks(self, x: str):
        # attack a single sentence to produce a batch
        originals = []
        attacked = []
        paraphrased = np.zeros((32))
        if self.attacks[0]:
            attacked = self.paraphraser.attack([x]) + [x for _ in range(16)]
            paraphrased[:4] = 1
        else:
            attacked = [x for _ in range(32)]
        originals = attacked.copy()

        char_masks = [np.zeros(len(x.split())) for x in attacked]
        concat_masks = [np.zeros(len(x.split())) for x in attacked]
        # a batch of 32 strings, 16 have been paraphrased, 16 are the original
        if self.attacks[1]:
            attacked, char_masks = self.char_attack.attack(attacked)
        if self.attacks[2]:
            attacked, concat_masks = self.concat_attack.attack(attacked)

        # ensure that char masks are long enough (i.e. have zeros where the concatenated words are)
        for i in range(len(char_masks)):
            diff = len(concat_masks[i]) - len(char_masks[i])
            char_masks[i] = np.append(char_masks[i], np.zeros((diff)))

        return attacked, (originals, paraphrased, char_masks, concat_masks)

    # Option1: fixed attempts
    def attack(self, model, test_sentences, test_labels, tokenizer, embedding):
        # test_set should be the set of test strings e.g. ['hello world', 'quick brown fox', ...]
        attacked_predictions = []
        model_preds = []

        for x in tqdm(test_sentences):
            y_prime = predict([x], model, tokenizer, embedding)[0]
            model_preds.append(y_prime)
            if y_prime != self.target_label:
                attacked, _ = self._gen_attacks(x)
                classes = predict(attacked, model, tokenizer, embedding)
                hit = classes.eq(self.target_label).sum() > 0
                if hit:
                    attacked_predictions.append(self.target_label)
                else:
                    attacked_predictions.append(y_prime)
            else:
                attacked_predictions.append(y_prime)

            logging.info(f"Fixed Attack: test sentence: {x}")
            for s in attacked:
                logging.info(f"Fixed Attack: {s}")

        attacked_predictions = np.array(attacked_predictions)
        model_preds = np.array(model_preds)
        # hit rate is all cases where model wasn't going to predict target but now does
        hit_rate = (
            attacked_predictions[model_preds != self.target_label] == self.target_label
        ).mean()
        results = {
            "new_acc": (attacked_predictions == test_labels).mean(),
            "min_acc": (test_labels == self.target_label).mean(),
            "hit_rate": hit_rate,
        }
        return results

        # print(f"Attack success rate: {100*hit_rate}%")
        # print(f"New acc: {(attacked_predictions==test_labels).mean()}")
        # print(
        #     f"Total prediction of target label: {(model_preds==self.target_label).mean()}"
        # )
        # print(
        #     f"True frequency of target label: {(test_labels==self.target_label).mean()}"
        # )
        # print(f"Model would predict the target for {hit_rate+}")

    def _breed(
        self, parents, originals, paraphrased, char_mask, concat_mask, fitnesses
    ):
        # how to combine parents?
        # case 1) one or both have been paraphrased -> select one of the parents by fitness prob.
        # case 2) neither have been paraphrased -> select word-by-word using fitness prob.
        p = torch.nn.functional.softmax(fitnesses, dim=0).detach().numpy()
        if paraphrased[0] == paraphrased[1] == 0:
            child_char_mask = np.zeros(len(char_mask[0]))
            child_concat_mask = np.zeros(len(concat_mask[0]))
            child_sentence = []
            for word_ind in range(len(parents[0].split())):
                parent_id = np.random.choice(2, p=p)
                child_char_mask[word_ind] = char_mask[parent_id][word_ind]
                child_concat_mask[word_ind] = concat_mask[parent_id][word_ind]
                child_sentence.append(parents[parent_id].split()[word_ind])

            return (
                " ".join(child_sentence),
                originals[0],
                False,
                child_char_mask,
                child_concat_mask,
            )
        else:
            ind = np.random.choice(2, p=p)
            return (
                parents[ind],
                originals[ind],
                paraphrased[ind],
                char_mask[ind].copy(),
                concat_mask[ind].copy(),
            )

    def _mutate(
        self,
        child_sentence,
        original,
        paraphrased,
        char_mask,
        concat_mask,
        num_mutations,
    ):
        # how to mutate whilst ensuring perturbation control.
        # pick a single word
        # -> if unattacked previously -> attack it and reset one of the character attacks at random
        # -> if it was character attacked -> reset it and perform a new character attack
        # -> if it was concat attacked -> remove it and replace with a new concat
        sent = child_sentence.split()
        for _ in range(num_mutations):
            mutate_ind = np.random.choice(len(sent))
            if not (char_mask[mutate_ind]) and not (concat_mask[mutate_ind]):
                if char_mask.sum() > 0 and len(sent) == len(char_mask):
                    reset_ind = np.random.choice(
                        len(sent), p=char_mask / char_mask.sum()
                    )
                    sent[reset_ind] = original.split()[reset_ind]
                    char_mask[reset_ind] = 0
                    char_mask[mutate_ind] = 1
                    sent[mutate_ind] = self.char_attack._attack(sent[mutate_ind])
            elif char_mask[mutate_ind] == 1:
                sent[mutate_ind] = self.char_attack._attack(
                    original.split()[mutate_ind]
                )
            elif concat_mask[mutate_ind]:
                attack = self.concat_attack._choose(1)
                sent[mutate_ind] = attack[0]
        return " ".join(sent), original, paraphrased, char_mask, concat_mask

    def _new_generation(
        self,
        generation,
        selection_probability,
        orig_sentences,
        paraphrased,
        char_masks,
        concat_masks,
        fitness,
        num_mutations,
    ):
        # update the masks (paraphrased, char_masks, concat_masks)
        new_generation = []
        new_originals = []
        new_paraphrased = np.zeros(32)
        new_char_masks = []
        new_concat_masks = []
        for gen_ind in range(len(generation)):

            # choose two parents for each sentence
            parents_indxs = np.random.choice(
                len(generation), p=selection_probability, size=2
            )
            #  parents, originals, paraphrased, char_mask, concat_mask, fitnesses
            (
                child,
                original,
                child_paraphrased,
                char_attack,
                concat_attack,
            ) = self._breed(
                [generation[i] for i in parents_indxs],
                [orig_sentences[i] for i in parents_indxs],
                paraphrased[parents_indxs],
                [char_masks[i] for i in parents_indxs],
                [concat_masks[i] for i in parents_indxs],
                fitness[parents_indxs],
            )
            logging.info(
                f"Genetic Attack: bred parents: {[generation[i] for i in parents_indxs]}"
            )
            logging.info(f"Genetic Attack: child: {child}")
            (
                child,
                original,
                child_paraphrased,
                char_attack,
                concat_attack,
            ) = self._mutate(
                child,
                original,
                child_paraphrased,
                char_attack.copy(),
                concat_attack.copy(),
                num_mutations=num_mutations,
            )
            logging.info(f"Genetic Attack: child mutated into: {child}")

            new_generation.append(child)
            new_char_masks.append(char_attack)
            new_concat_masks.append(concat_attack)
            new_originals.append(original)
            new_paraphrased[gen_ind] = child_paraphrased
        return (
            new_generation,
            new_originals,
            new_paraphrased,
            new_char_masks,
            new_concat_masks,
        )

    # Option2: adaptive/genetic algorithm
    def genetic_attack(
        self,
        model,
        surrogate_model,
        test_sentences,
        test_labels,
        tokenizer,
        embedding,
        surrogate_tokenizer,
        surrogate_embedding,
        max_generations=10,
        num_mutations=1,
        batch_size=128,
    ):
        model_preds = []
        attacked_predictions = np.zeros(len(test_labels))
        evaluations_per_sentence = []
        # batched approach: batch of generations => batch of elite members =>
        # only do this for cases where model doesn't predict y_prime as target
        for sentence_ind, sentence in enumerate(tqdm(test_sentences)):
            evaluations = 0
            y_prime = predict([sentence], model, tokenizer, embedding)[0]
            model_preds.append(y_prime)
            attacked_predictions[sentence_ind] = y_prime
            if y_prime != self.target_label:
                # create the first generation
                generation, (
                    orig_sentences,
                    paraphrased,
                    char_masks,
                    concat_masks,
                ) = self._gen_attacks(sentence)

                # use the surrogate model to compute fitness of each
                for _ in range(max_generations):
                    logits = forward(
                        generation,
                        surrogate_model,
                        surrogate_tokenizer,
                        surrogate_embedding,
                    )
                    logits = torch.log(
                        torch.nn.functional.softmax(logits, dim=1).detach()
                    )
                    logits *= -1
                    logits[:, self.target_label] *= -1
                    fitness = torch.sum(logits, dim=1)

                    # select elite member
                    elite_member = generation[torch.argmax(fitness)]

                    # check if elite member tricks the attacked model
                    pred = predict([elite_member], model, tokenizer, embedding)[0]
                    evaluations += 1

                    if pred == self.target_label:
                        attacked_predictions[sentence_ind] = self.target_label
                        logging.info(
                            f"Genetic Attack on {sentence} succeeded with {elite_member} after {evaluations} evals"
                        )
                        break

                    # if not, compute selection prob. using softmax of fitnesses
                    p = torch.nn.functional.softmax(fitness, dim=0).detach().numpy()

                    (
                        generation,
                        orig_sentences,
                        paraphrased,
                        char_masks,
                        concat_masks,
                    ) = self._new_generation(
                        generation=generation,
                        selection_probability=p,
                        orig_sentences=orig_sentences,
                        paraphrased=paraphrased,
                        char_masks=char_masks,
                        concat_masks=concat_masks,
                        fitness=fitness,
                        num_mutations=num_mutations,
                    )
            evaluations_per_sentence.append(evaluations)

        model_preds = np.array(model_preds)
        # hit rate is percentage of cases where model wasn't going to predict target but now does
        # hit_rate = (
        #     attacked_predictions[model_preds != self.target_label] == self.target_label
        # ).mean()
        results = {
            "evals_per_sentence": evaluations_per_sentence,
            "new_acc": (attacked_predictions == test_labels).mean(),
            "min_acc": (test_labels == self.target_label).mean(),
            "hit_rate": (
                attacked_predictions[model_preds != self.target_label]
                == self.target_label
            ).mean(),
            "model_preds": attacked_predictions,
        }
        return results
        # print(f"Attack success rate: {100*hit_rate}%")
        # print(f"New acc: {(attacked_predictions==test_labels).mean()}")
        # print(
        #     f"Total prediction of target label: {(model_preds==self.target_label).mean()}"
        # )
        # print(
        #     f"True frequency of target label: {(test_labels==self.target_label).mean()}"
        # )
        # print("MAX and MEAN evaluations per sentence")
        # print(max(evaluations_per_sentence))
        # print(sum(evaluations_per_sentence) / len(evaluations_per_sentence))


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
