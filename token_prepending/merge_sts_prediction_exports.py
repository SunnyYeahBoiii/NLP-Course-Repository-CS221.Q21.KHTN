import argparse

import pandas as pd


KEY_COLUMNS = ["task", "dataset", "row_index"]


def main():
    parser = argparse.ArgumentParser(
        description="Merge vanilla and TP pair-level STS prediction exports into one analysis CSV."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--task", default="", help="Optional task filter, e.g. STSBenchmark or SICKRelatedness.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.task:
        df = df[df["task"] == args.task].copy()

    vanilla = df[df["plan"] == "vanilla"].copy()
    tp = df[df["plan"] == "tp"].copy()

    if vanilla.empty:
        raise ValueError("No vanilla rows found in input CSV.")
    if tp.empty:
        raise ValueError("No TP rows found in input CSV.")

    vanilla = vanilla.rename(
        columns={
            "cosine": "vanilla_cosine",
            "pred_score_0_5": "vanilla_score_0_5",
            "abs_error_0_5": "abs_error_vanilla",
        }
    )
    tp = tp.rename(
        columns={
            "cosine": "tp_cosine",
            "pred_score_0_5": "tp_score_0_5",
            "abs_error_0_5": "abs_error_tp",
        }
    )

    merged = vanilla[
        KEY_COLUMNS
        + [
            "sentence1",
            "sentence2",
            "gold_score",
            "vanilla_cosine",
            "vanilla_score_0_5",
            "abs_error_vanilla",
        ]
    ].merge(
        tp[KEY_COLUMNS + ["tp_cosine", "tp_score_0_5", "abs_error_tp"]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    merged["error_gap"] = merged["abs_error_tp"] - merged["abs_error_vanilla"]
    merged["tp_better"] = merged["error_gap"] < 0
    merged.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(merged)} rows to {args.output_csv}")
    print("Largest TP failures:")
    print(
        merged.sort_values("error_gap", ascending=False)
        .head(5)[KEY_COLUMNS + ["gold_score", "vanilla_score_0_5", "tp_score_0_5", "error_gap"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
