import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vietnamese_sts import cosine_similarity_scores, preprocess_sentence_for_prompt


class FakeViTokenizer:
    @staticmethod
    def tokenize(text):
        if text == "Truong dai hoc Bach Khoa Ha Noi":
            return "Truong_dai_hoc Bach_Khoa Ha_Noi"
        return text.replace("nguoi dan ong", "nguoi_dan_ong")


class VietnameseStsTest(unittest.TestCase):
    def test_preprocess_uses_pyvi_before_prompt_punctuation(self):
        sentence = preprocess_sentence_for_prompt(
            "Truong dai hoc Bach Khoa Ha Noi",
            vi_tokenizer=FakeViTokenizer,
        )

        self.assertEqual(sentence, "Truong_dai_hoc Bach_Khoa Ha_Noi.")

    def test_preprocess_converts_question_mark_after_pyvi(self):
        sentence = preprocess_sentence_for_prompt(
            "nguoi dan ong dang chay?",
            vi_tokenizer=FakeViTokenizer,
        )

        self.assertEqual(sentence, "nguoi_dan_ong dang chay.")

    def test_cosine_similarity_scores(self):
        scores = cosine_similarity_scores(
            np.array([[1.0, 0.0], [1.0, 1.0]]),
            np.array([[1.0, 0.0], [1.0, -1.0]]),
        )

        np.testing.assert_allclose(scores, np.array([1.0, 0.0]), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
