from typing import Iterable, Optional, Tuple

import numpy as np


DEFAULT_VIETNAMESE_STS_DATASET = "nemixo/stsbenchmark-sts-vietnamese"
DEFAULT_VIETNAMESE_STS_SPLIT = "test"


def preprocess_sentence_for_prompt(text, use_vi_tokenizer: bool = True, vi_tokenizer=None) -> str:
    sentence = "" if text is None else str(text)
    if use_vi_tokenizer:
        if vi_tokenizer is None:
            from pyvi import ViTokenizer

            vi_tokenizer = ViTokenizer
        sentence = vi_tokenizer.tokenize(sentence)

    if sentence and sentence[-1] not in ".?\"'":
        sentence += "."
    sentence = sentence.replace('"', "'")
    if sentence and sentence[-1] == "?":
        sentence = sentence[:-1] + "."
    return sentence


def cosine_similarity_scores(embeddings_a, embeddings_b) -> np.ndarray:
    a = np.asarray(embeddings_a, dtype=np.float64)
    b = np.asarray(embeddings_b, dtype=np.float64)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def resolve_sts_columns(column_names: Iterable[str]) -> Tuple[str, str, str]:
    names = set(column_names)
    candidates = [
        ("sentence1", "sentence2", "score"),
        ("sentence1", "sentence2", "label"),
        ("sent1", "sent2", "score"),
        ("text1", "text2", "score"),
    ]
    for sentence1, sentence2, score in candidates:
        if {sentence1, sentence2, score}.issubset(names):
            return sentence1, sentence2, score
    raise ValueError(
        "Cannot infer STS columns. Expected one of: "
        "sentence1/sentence2/score, sentence1/sentence2/label, "
        "sent1/sent2/score, text1/text2/score."
    )


def select_dataset_split(dataset, split_name: Optional[str]):
    if not hasattr(dataset, "keys"):
        return dataset

    if split_name in dataset:
        return dataset[split_name]

    available_splits = list(dataset.keys())
    preferred_splits = ("test", "validation", "dev", "train")
    for fallback in preferred_splits:
        if fallback in dataset:
            print(
                f"warning: split '{split_name}' not found; using '{fallback}'. "
                f"available splits: {available_splits}"
            )
            return dataset[fallback]

    raise ValueError(
        f"split '{split_name}' not found. available splits: {available_splits}"
    )
