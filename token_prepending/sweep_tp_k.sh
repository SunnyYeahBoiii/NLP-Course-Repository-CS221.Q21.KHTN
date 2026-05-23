#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/workspace/llm_weights/Llama-2-7b-hf"
BATCH_SIZE=16
OUTPUT_LAYER=27
TP_START=1
PROMPT_METHOD="prompteol"
OUT_DIR="sweep_tp_k_llama2_7b_${PROMPT_METHOD}_stsb"

mkdir -p "${OUT_DIR}/logs"

CSV_FILE="${OUT_DIR}/results.csv"
echo "k,STSBenchmark" > "${CSV_FILE}"

for K in 5 6 7 8 9 10; do
LOG_FILE="${OUT_DIR}/logs/k_${K}.log"

echo "Running tp_exiting_index=${K}"

python evaluate.py \
    --model_name_or_path "${MODEL_PATH}" \
    --use_which_plan tp \
    --output_layer "${OUTPUT_LAYER}" \
    --tp_starting_index "${TP_START}" \
    --tp_exiting_index "${K}" \
    --batch_size "${BATCH_SIZE}" \
    --mode test \
    --task_set stsb \
    --prompt_method "${PROMPT_METHOD}" \
    2>&1 | tee "${LOG_FILE}"

python - "${LOG_FILE}" "${K}" >> "${CSV_FILE}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
k = sys.argv[2]
text = log_path.read_text(encoding="utf-8", errors="ignore")

match = re.search(r"test\s*:\s*pearson\s*=\s*([0-9.]+),\s*spearman\s*=\s*([0-9.]+)", text)
if not match:
    raise SystemExit(f"Could not find STSBenchmark score in {log_path}")

stsb = float(match.group(2)) * 100
print(f"{k},{stsb:.2f}")
PY
done

echo
echo "Done. CSV saved to: ${CSV_FILE}"

python - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sweep_tp_k_llama2_7b_prompteol_stsb/results.csv")

plt.figure(figsize=(8, 5))
plt.plot(df["k"], df["STSBenchmark"], marker="o")
plt.xlabel("tp_exiting_index (k)")
plt.ylabel("STSBenchmark")
plt.title("Llama-2-7B PromptEOL + TP sweep over k")
plt.xticks(df["k"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sweep_tp_k_llama2_7b_prompteol_stsb/linechart.png", dpi=200)
print("Saved to sweep_tp_k_llama2_7b_prompteol_stsb/linechart.png")
PY