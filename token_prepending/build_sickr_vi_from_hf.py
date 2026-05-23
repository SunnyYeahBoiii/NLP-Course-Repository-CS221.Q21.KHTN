# conda create -n data-prep python=3.10 -y
# conda activate data-prep
# pip install -U datasets pandas tqdm

import argparse
import csv
from math import isclose

from datasets import load_dataset
from tqdm import tqdm


DEFAULT_SICK_TEST = "SentEval/data/downstream/SICK/SICK_test_annotated.txt"
DEFAULT_OUTPUT = "SentEval/data/downstream/SICK/SICK_test_annotated_vi.txt"
DEFAULT_DATASET = "GreenNode/sickr-sts-vn"
DEFAULT_SUBSET = "default"
DEFAULT_SPLIT = "test"


def load_sick_test(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), reader.fieldnames


def load_vietnamese_rows(dataset_name, subset, split):
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return list(dataset)


def validate_sick_columns(fieldnames):
    required_columns = {
        "pair_ID",
        "sentence_A",
        "sentence_B",
        "relatedness_score",
        "entailment_judgment",
    }
    missing_columns = required_columns - set(fieldnames or [])
    if missing_columns:
        raise ValueError(f"Missing required SICK columns: {sorted(missing_columns)}")


def build_vietnamese_sick_test(sick_rows, hf_rows, validate_score):
    if len(hf_rows) < len(sick_rows):
        raise ValueError(
            f"HF dataset has fewer rows ({len(hf_rows)}) than SICK test ({len(sick_rows)})"
        )

    offset = len(hf_rows) - len(sick_rows)
    hf_test_rows = hf_rows[offset:]
    converted = []
    score_mismatches = []

    for sick_row, hf_row in tqdm(
        zip(sick_rows, hf_test_rows),
        total=len(sick_rows),
        desc="Building Vietnamese SICK-R test file from HF tail",
    ):
        sick_score = float(sick_row["relatedness_score"])
        hf_score = float(hf_row["score"])

        if validate_score and not isclose(sick_score, hf_score, rel_tol=0.0, abs_tol=1e-6):
            score_mismatches.append(
                {
                    "pair_ID": sick_row["pair_ID"],
                    "sick_score": sick_score,
                    "hf_score": hf_score,
                    "sick_sentence_A": sick_row["sentence_A"],
                    "sick_sentence_B": sick_row["sentence_B"],
                    "hf_og_sentence1": hf_row.get("og_sentence1", ""),
                    "hf_og_sentence2": hf_row.get("og_sentence2", ""),
                }
            )
            continue

        converted.append(
            {
                "pair_ID": sick_row["pair_ID"],
                "sentence_A": hf_row["sentence1"],
                "sentence_B": hf_row["sentence2"],
                "relatedness_score": sick_row["relatedness_score"],
                "entailment_judgment": sick_row["entailment_judgment"],
            }
        )

    return converted, score_mismatches, offset


def write_sick_file(path, rows):
    fieldnames = [
        "pair_ID",
        "sentence_A",
        "sentence_B",
        "relatedness_score",
        "entailment_judgment",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_mismatch_report(path, mismatches):
    fieldnames = [
        "pair_ID",
        "sick_score",
        "hf_score",
        "sick_sentence_A",
        "sick_sentence_B",
        "hf_og_sentence1",
        "hf_og_sentence2",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(mismatches)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build SentEval SICK_test_annotated_vi.txt from the tail split of "
            "GreenNode/sickr-sts-vn. Assumes HF rows are ordered as train + trial + test."
        )
    )
    parser.add_argument("--sick_test", default=DEFAULT_SICK_TEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--no_validate_score", action="store_true")
    parser.add_argument("--mismatch_report", default="sickr_vi_score_mismatches.tsv")
    args = parser.parse_args()

    sick_rows, fieldnames = load_sick_test(args.sick_test)
    validate_sick_columns(fieldnames)

    hf_rows = load_vietnamese_rows(args.dataset, args.subset, args.split)
    converted, mismatches, offset = build_vietnamese_sick_test(
        sick_rows,
        hf_rows,
        validate_score=not args.no_validate_score,
    )

    if mismatches:
        write_mismatch_report(args.mismatch_report, mismatches)
        raise RuntimeError(
            f"Found {len(mismatches)} score mismatches. "
            f"Report written to {args.mismatch_report}. Output file was not written."
        )

    if len(converted) != len(sick_rows):
        raise RuntimeError(
            f"Converted {len(converted)} rows, expected {len(sick_rows)} rows."
        )

    write_sick_file(args.output, converted)
    print(f"HF rows: {len(hf_rows)}")
    print(f"SICK test rows: {len(sick_rows)}")
    print(f"Tail offset: {offset}")
    print(f"Wrote {len(converted)} rows to {args.output}")


if __name__ == "__main__":
    main()
