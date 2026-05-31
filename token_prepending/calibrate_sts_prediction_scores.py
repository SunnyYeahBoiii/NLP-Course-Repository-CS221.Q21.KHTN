import argparse

import pandas as pd


def minmax_score(series):
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(2.5, index=series.index)
    return 5.0 * (series - min_value) / (max_value - min_value)


def calibrate_long(df):
    required = {"task", "dataset", "plan", "cosine", "gold_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required long-format columns: {sorted(missing)}")

    group_columns = ["task", "dataset", "plan"]
    df["calibrated_score_0_5"] = df.groupby(group_columns)["cosine"].transform(minmax_score)
    df["calibrated_abs_error"] = (df["calibrated_score_0_5"] - df["gold_score"]).abs()
    return df


def calibrate_wide(df):
    required = {"task", "dataset", "vanilla_cosine", "tp_cosine", "gold_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required wide-format columns: {sorted(missing)}")

    group_columns = ["task", "dataset"]
    df["vanilla_calibrated_score_0_5"] = df.groupby(group_columns)["vanilla_cosine"].transform(minmax_score)
    df["tp_calibrated_score_0_5"] = df.groupby(group_columns)["tp_cosine"].transform(minmax_score)
    df["calibrated_abs_error_vanilla"] = (df["vanilla_calibrated_score_0_5"] - df["gold_score"]).abs()
    df["calibrated_abs_error_tp"] = (df["tp_calibrated_score_0_5"] - df["gold_score"]).abs()
    df["calibrated_error_gap"] = df["calibrated_abs_error_tp"] - df["calibrated_abs_error_vanilla"]
    df["tp_better_calibrated"] = df["calibrated_error_gap"] < 0
    return df


def infer_format(df):
    if {"plan", "cosine"}.issubset(df.columns):
        return "long"
    if {"vanilla_cosine", "tp_cosine"}.issubset(df.columns):
        return "wide"
    raise ValueError("Could not infer CSV format. Use --format long or --format wide.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw cosine values to task/model-calibrated 0-5 scores for STS analysis."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--format", choices=["auto", "long", "wide"], default="auto")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    csv_format = infer_format(df) if args.format == "auto" else args.format

    if csv_format == "long":
        calibrated = calibrate_long(df)
    else:
        calibrated = calibrate_wide(df)

    calibrated.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(calibrated)} rows to {args.output_csv}")
    print(f"Input format: {csv_format}")

    if csv_format == "wide":
        print("Largest TP failures after calibration:")
        print(
            calibrated.sort_values("calibrated_error_gap", ascending=False)
            .head(5)[
                [
                    "task",
                    "dataset",
                    "row_index",
                    "gold_score",
                    "vanilla_calibrated_score_0_5",
                    "tp_calibrated_score_0_5",
                    "calibrated_error_gap",
                ]
            ]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
