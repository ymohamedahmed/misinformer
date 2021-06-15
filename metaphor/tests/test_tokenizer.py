import unittest
from metaphor.models.common.tokenize import Tokenizer
import torch


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_dictionary(self):
        test_sentences = ["Hello world", "Hello", ""]
        tokenized = self.tokenizer(test_sentences)
        self.assertTrue(
            torch.all(
                torch.eq(tokenized, torch.Tensor([[2.0, 1.0], [2.0, 0.0], [0.0, 0.0]]))
            )
        )
        self.assertTrue(
            torch.all(self.tokenizer.sentence_lengths.eq(torch.Tensor([2, 1, 0])))
        )
        expected_padded = ["hello world", "hello <PAD>", "<PAD> <PAD>"]
        self.assertTrue(
            all(
                [
                    self.tokenizer.padded_sentences[i] == expected_padded[i]
                    for i in range(len(expected_padded))
                ]
            )
        )

        self.assertTrue(
            torch.all(
                torch.eq(self.tokenizer.mask, torch.tensor([[1, 1], [1, 0], [0, 0]]))
            )
        )


if __name__ == "__main__":
    unittest.main()
