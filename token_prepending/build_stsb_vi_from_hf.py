import argparse
import csv
from math import isclose

from datasets import load_dataset
from tqdm import tqdm


DEFAULT_STSB_TEST = "SentEval/data/downstream/STS/STSBenchmark/sts-test.csv"
DEFAULT_OUTPUT = "SentEval/data/downstream/STS/STSBenchmark/sts-test-vi.csv"
DEFAULT_DATASET = "GreenNode/stsbenchmark-sts-vn"
DEFAULT_SUBSET = "default"
DEFAULT_SPLIT = "test"


def clean_text(text):
    return " ".join(
        str(text)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )


def load_stsb_test(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            parts = line.split("\t", 5)
            if len(parts) != 6:
                raise ValueError(
                    f"Invalid STS-B row at line {line_number}: expected at least 6 tab-separated parts, got {len(parts)}"
                )

            sentence_parts = parts[5].rsplit("\t", 1)
            if len(sentence_parts) != 2:
                raise ValueError(
                    f"Invalid STS-B row at line {line_number}: could not split sentence pair"
                )

            rows.append(parts[:5] + sentence_parts)
    return rows


def load_vietnamese_rows(dataset_name, subset, split):
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return list(dataset)


def find_column(row, candidates):
    for name in candidates:
        if name in row:
            return name
    raise KeyError(f"Could not find any columns from: {candidates}")


def build_vietnamese_stsb_test(stsb_rows, hf_rows, validate_score):
    if len(stsb_rows) != len(hf_rows):
        raise ValueError(
            f"STS-B test has {len(stsb_rows)} rows but HF split has {len(hf_rows)} rows"
        )

    if not hf_rows:
        raise ValueError("HF split is empty")

    sentence1_col = find_column(hf_rows[0], ["sentence1", "sentence_A", "sent1"])
    sentence2_col = find_column(hf_rows[0], ["sentence2", "sentence_B", "sent2"])
    score_col = find_column(hf_rows[0], ["score", "label", "similarity_score"])

    converted = []
    score_mismatches = []

    for index, (stsb_row, hf_row) in enumerate(
        tqdm(
            zip(stsb_rows, hf_rows),
            total=len(stsb_rows),
            desc="Building Vietnamese STS-B test file",
        )
    ):
        stsb_score = float(stsb_row[4])
        hf_score = float(hf_row[score_col])

        if validate_score and not isclose(stsb_score, hf_score, rel_tol=0.0, abs_tol=1e-6):
            score_mismatches.append(
                {
                    "row_index": index,
                    "stsb_score": stsb_score,
                    "hf_score": hf_score,
                    "stsb_sentence1": stsb_row[5],
                    "stsb_sentence2": stsb_row[6],
                    "hf_sentence1": hf_row[sentence1_col],
                    "hf_sentence2": hf_row[sentence2_col],
                }
            )
            continue

        new_row = list(stsb_row)
        new_row[5] = clean_text(hf_row[sentence1_col])
        new_row[6] = clean_text(hf_row[sentence2_col])
        converted.append(new_row)

    return converted, score_mismatches


def write_stsb_file(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def write_mismatch_report(path, mismatches):
    fieldnames = [
        "row_index",
        "stsb_score",
        "hf_score",
        "stsb_sentence1",
        "stsb_sentence2",
        "hf_sentence1",
        "hf_sentence2",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(mismatches)


def main():
    parser = argparse.ArgumentParser(
        description="Build a Vietnamese SentEval sts-test.csv from GreenNode/stsbenchmark-sts-vn test split."
    )
    parser.add_argument("--stsb_test", default=DEFAULT_STSB_TEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--no_validate_score", action="store_true")
    parser.add_argument("--mismatch_report", default="stsb_vi_score_mismatches.tsv")
    args = parser.parse_args()

    stsb_rows = load_stsb_test(args.stsb_test)
    hf_rows = load_vietnamese_rows(args.dataset, args.subset, args.split)
    converted, mismatches = build_vietnamese_stsb_test(
        stsb_rows,
        hf_rows,
        validate_score=not args.no_validate_score,
    )

    if mismatches:
        write_mismatch_report(args.mismatch_report, mismatches)
        raise RuntimeError(
            f"Found {len(mismatches)} score mismatches. "
            f"Report written to {args.mismatch_report}. Output file was not written."
        )

    if len(converted) != len(stsb_rows):
        raise RuntimeError(
            f"Converted {len(converted)} rows, expected {len(stsb_rows)} rows."
        )

    write_stsb_file(args.output, converted)
    print(f"HF rows: {len(hf_rows)}")
    print(f"STS-B test rows: {len(stsb_rows)}")
    print(f"Wrote {len(converted)} rows to {args.output}")


if __name__ == "__main__":
    main()
