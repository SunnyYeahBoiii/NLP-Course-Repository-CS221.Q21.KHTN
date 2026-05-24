import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "vinai/phobert-base-v2"
DEFAULT_STSB_PATH = "SentEval/data/downstream/STS/STSBenchmark/sts-test-vi.csv"
DEFAULT_SICKR_PATH = "SentEval/data/downstream/SICK/SICK_test_annotated_vi.txt"
DEFAULT_PROMPT_TEMPLATE = 'This sentence: "*sent 0*" means <mask>'


def clean_text(text):
    return " ".join(str(text).replace("\t", " ").replace("\n", " ").split())


def load_stsb(path):
    examples = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 7:
                raise ValueError(
                    f"Invalid STS-B row at line {line_number}: expected 7 columns, got {len(row)}"
                )
            examples.append(
                {
                    "sentence1": clean_text(row[5]),
                    "sentence2": clean_text(row[6]),
                    "score": float(row[4]),
                }
            )
    return examples


def load_sickr(path):
    examples = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sentence_A", "sentence_B", "relatedness_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing SICK-R columns: {sorted(missing)}")

        for row in reader:
            examples.append(
                {
                    "sentence1": clean_text(row["sentence_A"]),
                    "sentence2": clean_text(row["sentence_B"]),
                    "score": float(row["relatedness_score"]),
                }
            )
    return examples


def get_word_segmenter(name):
    if name == "none":
        return lambda text: text
    if name == "underthesea":
        from underthesea import word_tokenize

        return lambda text: word_tokenize(text, format="text")
    if name == "pyvi":
        from pyvi.ViTokenizer import tokenize

        return tokenize
    raise ValueError(f"Unsupported word segmenter: {name}")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def mask_pool(last_hidden_state, input_ids, mask_token_id):
    mask_positions = input_ids.eq(mask_token_id)
    if not mask_positions.any(dim=1).all():
        raise ValueError("Every prompted sentence must contain the tokenizer mask token.")

    first_mask_indices = mask_positions.float().argmax(dim=1)
    batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
    return last_hidden_state[batch_indices, first_mask_indices, :]


def encode_sentences(
    sentences,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
    pooling,
    word_segmenter,
    encode_mode,
    prompt_template,
):
    embeddings = []
    for start in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
        batch_sentences = []
        for sentence in sentences[start : start + batch_size]:
            segmented_sentence = word_segmenter(sentence)
            if encode_mode == "avg":
                batch_sentences.append(segmented_sentence)
            elif encode_mode == "prompt":
                batch_sentences.append(prompt_template.replace("*sent 0*", segmented_sentence))
            else:
                raise ValueError(f"Unsupported encode mode: {encode_mode}")

        batch = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = model(**batch, return_dict=True)
            if encode_mode == "prompt":
                pooled = mask_pool(
                    outputs.last_hidden_state,
                    batch["input_ids"],
                    tokenizer.mask_token_id,
                )
            elif pooling == "cls":
                pooled = outputs.last_hidden_state[:, 0, :]
            elif pooling == "mean":
                pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")

            pooled = F.normalize(pooled.float(), p=2, dim=1)
            embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0).numpy()


def evaluate_examples(
    name,
    examples,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
    pooling,
    word_segmenter,
    encode_mode,
    prompt_template,
):
    sentence1 = [example["sentence1"] for example in examples]
    sentence2 = [example["sentence2"] for example in examples]
    labels = np.array([example["score"] for example in examples], dtype=np.float64)

    emb1 = encode_sentences(
        sentence1,
        tokenizer,
        model,
        device,
        batch_size,
        max_length,
        pooling,
        word_segmenter,
        encode_mode,
        prompt_template,
    )
    emb2 = encode_sentences(
        sentence2,
        tokenizer,
        model,
        device,
        batch_size,
        max_length,
        pooling,
        word_segmenter,
        encode_mode,
        prompt_template,
    )

    similarities = (emb1 * emb2).sum(axis=1)
    pearson = pearsonr(similarities, labels)[0]
    spearman = spearmanr(similarities, labels)[0]

    print(f"{name}:")
    print(f"  rows     : {len(examples)}")
    print(f"  pearson  : {pearson:.4f} ({pearson * 100:.2f})")
    print(f"  spearman : {spearman:.4f} ({spearman * 100:.2f})")

    return {
        "task": name,
        "rows": len(examples),
        "pearson": pearson,
        "spearman": spearman,
    }


def write_results(
    path,
    model_name,
    encode_mode,
    pooling,
    word_segmenter_name,
    prompt_template,
    results,
):
    exists = Path(path).exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "encode_mode",
            "pooling",
            "word_segmenter",
            "prompt_template",
            "task",
            "rows",
            "pearson",
            "spearman",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": model_name,
                    "encode_mode": encode_mode,
                    "pooling": pooling,
                    "word_segmenter": word_segmenter_name,
                    "prompt_template": prompt_template,
                    "task": result["task"],
                    "rows": result["rows"],
                    "pearson": f"{result['pearson']:.6f}",
                    "spearman": f"{result['spearman']:.6f}",
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PhoBERT embeddings on Vietnamese STS-B and SICK-R SentEval files."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--stsb_path", default=DEFAULT_STSB_PATH)
    parser.add_argument("--sickr_path", default=DEFAULT_SICKR_PATH)
    parser.add_argument("--task", choices=["stsb", "sickr", "all"], default="all")
    parser.add_argument("--encode_mode", choices=["avg", "prompt"], default="avg")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--word_segmenter", choices=["none", "underthesea", "pyvi"], default="none")
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Encode mode: {args.encode_mode}")
    print(f"Pooling: {args.pooling}")
    print(f"Word segmenter: {args.word_segmenter}")
    if args.encode_mode == "prompt":
        print(f"Prompt template: {args.prompt_template}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    if args.encode_mode == "prompt":
        print(f"Tokenizer mask token: {tokenizer.mask_token}")
        print(f"Tokenizer mask token id: {tokenizer.mask_token_id}")
        if tokenizer.mask_token not in args.prompt_template:
            raise ValueError(
                f"Prompt template must contain the tokenizer mask token "
                f"{tokenizer.mask_token!r}, got: {args.prompt_template!r}"
            )

    word_segmenter = get_word_segmenter(args.word_segmenter)

    results = []
    if args.task in {"stsb", "all"}:
        stsb_examples = load_stsb(args.stsb_path)
        results.append(
            evaluate_examples(
                "STSBenchmark-Vi",
                stsb_examples,
                tokenizer,
                model,
                device,
                args.batch_size,
                args.max_length,
                args.pooling,
                word_segmenter,
                args.encode_mode,
                args.prompt_template,
            )
        )

    if args.task in {"sickr", "all"}:
        sickr_examples = load_sickr(args.sickr_path)
        results.append(
            evaluate_examples(
                "SICKRelatedness-Vi",
                sickr_examples,
                tokenizer,
                model,
                device,
                args.batch_size,
                args.max_length,
                args.pooling,
                word_segmenter,
                args.encode_mode,
                args.prompt_template,
            )
        )

    if args.output_csv:
        write_results(
            args.output_csv,
            args.model_name_or_path,
            args.encode_mode,
            args.pooling,
            args.word_segmenter,
            args.prompt_template,
            results,
        )
        print(f"Wrote results to {args.output_csv}")


if __name__ == "__main__":
    main()
