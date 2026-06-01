# CS221 Token Prepending

- Reproduce Token Prepending trên SentEval tiếng Anh.
- Sweep `tp_exiting_index = k` cho PromptEOL + TP.
- Tạo và chạy STS-B-Vi, SICK-R-Vi.
- Chạy baseline PhoBERT trên dữ liệu tiếng Việt.

Code chính nằm ở [`token_prepending`](token_prepending).

## 1. Môi Trường

Khuyến nghị:

- Linux GPU server.
- GPU: RTX 3090 24GB VRAM hoặc tương đương.
- RAM: 32GB+.
- Disk trống: 80GB+ nếu tải cả LLaMA2-7B và Qwen2-7B.
- Python: 3.9 cho môi trường chạy model.

```bash
cd /workspace
git clone -b prompteol-experiment https://github.com/SunnyYeahBoiii/NLP-Course-Repository-CS221.Q21.KHTN.git CS221
cd /workspace/CS221/token_prepending

conda create -n tp python=3.9 -y
conda activate tp
pip install -r requirements.txt
pip install "huggingface-hub>=0.23.2,<1.0"
```

Nếu bị OOM:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Sau đó giảm `batch_size` trong config.

## 2. Tải SentEval

```bash
cd /workspace/CS221/token_prepending
cd SentEval/data/downstream
bash download_dataset.sh
cd ../../..
```

Các file cần có:

```text
SentEval/data/downstream/STS/STSBenchmark/sts-test.csv
SentEval/data/downstream/SICK/SICK_test_annotated.txt
```

## 3. Tải Model Hugging Face

LLaMA2 cần tài khoản Hugging Face đã được cấp quyền.

```bash
hf auth login

mkdir -p /workspace/llm_weights

hf download meta-llama/Llama-2-7b-hf \
  --local-dir /workspace/llm_weights/Llama-2-7b-hf

hf download Qwen/Qwen2-7B \
  --local-dir /workspace/llm_weights/Qwen2-7B
```

Kiểm tra:

```bash
df -h /workspace
du -sh /workspace/llm_weights/*
```

## 4. Reproduce Paper Trên SentEval Tiếng Anh

Tạo config:

```bash
cat > config_reproduce_en.yaml <<'YAML'
default_config: llama-2-7b-prompteol-vanilla-en

gpu_config:
  cuda_visible_devices: "0,1"

models:
  llama-2-7b-prompteol-vanilla-en:
    model_name_or_path: "/workspace/llm_weights/Llama-2-7b-hf"
    use_which_plan: vanilla
    output_layer: -1
    tp_starting_index: 0
    tp_exiting_index: 0
    batch_size: 16
    mode: test
    task_set: sts
    prompt_method: prompteol
    prompt_language: en

  llama-2-7b-prompteol-tp-en:
    model_name_or_path: "/workspace/llm_weights/Llama-2-7b-hf"
    use_which_plan: tp
    output_layer: 27
    tp_starting_index: 1
    tp_exiting_index: 7
    batch_size: 16
    mode: test
    task_set: sts
    prompt_method: prompteol
    prompt_language: en

  llama-2-7b-cot-vanilla-en:
    model_name_or_path: "/workspace/llm_weights/Llama-2-7b-hf"
    use_which_plan: vanilla
    output_layer: -2
    tp_starting_index: 0
    tp_exiting_index: 0
    batch_size: 16
    mode: test
    task_set: sts
    prompt_method: cot
    prompt_language: en

  llama-2-7b-cot-tp-en:
    model_name_or_path: "/workspace/llm_weights/Llama-2-7b-hf"
    use_which_plan: tp
    output_layer: 27
    tp_starting_index: 1
    tp_exiting_index: 7
    batch_size: 16
    mode: test
    task_set: sts
    prompt_method: cot
    prompt_language: en
YAML
```

Chạy:

```bash
bash run.sh llama-2-7b-prompteol-vanilla-en config_reproduce_en.yaml
bash run.sh llama-2-7b-prompteol-tp-en config_reproduce_en.yaml
bash run.sh llama-2-7b-cot-vanilla-en config_reproduce_en.yaml
bash run.sh llama-2-7b-cot-tp-en config_reproduce_en.yaml
```

Ghi chú: với LLaMA2-7B CoT vanilla, dùng `output_layer: -2`. Khi dùng `-1`, STS-B/SICK-R có thể thấp bất thường.

## 5. Sweep `tp_exiting_index`

Script có sẵn chạy `k = 5..10` trên STSBenchmark:

```bash
bash sweep_tp_k.sh
```

Output:

```text
sweep_tp_k_llama2_7b_prompteol_stsb/results.csv
sweep_tp_k_llama2_7b_prompteol_stsb/linechart.png
```

## 6. Tạo Dữ Liệu Tiếng Việt

Tạo môi trường riêng cho data preparation:

```bash
conda create -n data-prep python=3.10 -y
conda activate data-prep
pip install -U datasets pandas tqdm pyarrow
cd /workspace/CS221/token_prepending
```

Build SICK-R-Vi:

```bash
python build_sickr_vi_from_hf.py
```

Output:

```text
SentEval/data/downstream/SICK/SICK_test_annotated_vi.txt
```

Build STS-B-Vi:

```bash
python build_stsb_vi_from_hf.py
```

Output:

```text
SentEval/data/downstream/STS/STSBenchmark/sts-test-vi.csv
```

Để SentEval dùng bản tiếng Việt:

```bash
cp SentEval/data/downstream/STS/STSBenchmark/sts-test.csv SentEval/data/downstream/STS/STSBenchmark/sts-test.en.csv
cp SentEval/data/downstream/SICK/SICK_test_annotated.txt SentEval/data/downstream/SICK/SICK_test_annotated.en.txt

cp SentEval/data/downstream/STS/STSBenchmark/sts-test-vi.csv SentEval/data/downstream/STS/STSBenchmark/sts-test.csv
cp SentEval/data/downstream/SICK/SICK_test_annotated_vi.txt SentEval/data/downstream/SICK/SICK_test_annotated.txt
```

Khôi phục tiếng Anh:

```bash
cp SentEval/data/downstream/STS/STSBenchmark/sts-test.en.csv SentEval/data/downstream/STS/STSBenchmark/sts-test.csv
cp SentEval/data/downstream/SICK/SICK_test_annotated.en.txt SentEval/data/downstream/SICK/SICK_test_annotated.txt
```

## 7. Chạy LLaMA2/Qwen2 Trên Data Tiếng Việt

Tạo config English prompt:

```bash
cat > config_reproduce_vi_en_prompt.yaml <<'YAML'
default_config: llama-2-7b-vi-vanilla-en-prompt

gpu_config:
  cuda_visible_devices: "0,1"

models:
  llama-2-7b-vi-vanilla-en-prompt:
    model_name_or_path: "/workspace/llm_weights/Llama-2-7b-hf"
    use_which_plan: vanilla
    output_layer: -1
    tp_starting_index: 0
    tp_exiting_index: 0
    batch_size: 8
    mode: test
    task_set: sts
    prompt_method: prompteol
    prompt_language: en

  llama-2-7b-vi-tp-en-prompt:
    model_name_or_path: "/workspace/llm_weights/Llama-2-7b-hf"
    use_which_plan: tp
    output_layer: 27
    tp_starting_index: 1
    tp_exiting_index: 7
    batch_size: 8
    mode: test
    task_set: sts
    prompt_method: prompteol
    prompt_language: en

  qwen2-7b-vi-vanilla-en-prompt:
    model_name_or_path: "/workspace/llm_weights/Qwen2-7B"
    use_which_plan: vanilla
    output_layer: -1
    tp_starting_index: 0
    tp_exiting_index: 0
    batch_size: 16
    mode: test
    task_set: sts
    prompt_method: prompteol
    prompt_language: en

  qwen2-7b-vi-tp-en-prompt:
    model_name_or_path: "/workspace/llm_weights/Qwen2-7B"
    use_which_plan: tp
    output_layer: -2
    tp_starting_index: 1
    tp_exiting_index: 6
    batch_size: 16
    mode: test
    task_set: sts
    prompt_method: prompteol
    prompt_language: en
YAML
```

Chạy:

```bash
conda activate tp
cd /workspace/CS221/token_prepending

bash run.sh llama-2-7b-vi-vanilla-en-prompt config_reproduce_vi_en_prompt.yaml
bash run.sh llama-2-7b-vi-tp-en-prompt config_reproduce_vi_en_prompt.yaml
bash run.sh qwen2-7b-vi-vanilla-en-prompt config_reproduce_vi_en_prompt.yaml
bash run.sh qwen2-7b-vi-tp-en-prompt config_reproduce_vi_en_prompt.yaml
```

Export tung sample de phan tich loi:

```bash
mkdir -p analysis

bash run.sh qwen2-7b-vi-vanilla-en-prompt config_reproduce_vi_en_prompt.yaml \
  --prediction_output_csv analysis/qwen2_vi_predictions_long.csv

bash run.sh qwen2-7b-vi-tp-en-prompt config_reproduce_vi_en_prompt.yaml \
  --prediction_output_csv analysis/qwen2_vi_predictions_long.csv

python merge_sts_prediction_exports.py \
  --input_csv analysis/qwen2_vi_predictions_long.csv \
  --output_csv analysis/qwen2_vi_predictions_wide.csv
```

File `analysis/qwen2_vi_predictions_wide.csv` co cac cot chinh: `sentence1`, `sentence2`, `gold_score`, `vanilla_cosine`, `tp_cosine`, `abs_error_vanilla`, `abs_error_tp`, `error_gap`.

Kết quả đã ghi nhận:

| Dataset | Model | Plan | Spearman |
|---|---|---|---:|
| SICK-R-Vi | LLaMA2-7B | vanilla | 60.62 |
| SICK-R-Vi | LLaMA2-7B | TP | 67.19 |
| STS-B-Vi | LLaMA2-7B | vanilla | 59.49 |
| STS-B-Vi | LLaMA2-7B | TP | 67.87 |
| STS-B-Vi | Qwen2-7B | vanilla | 65.53 |
| SICK-R-Vi | Qwen2-7B | vanilla | 68.47 |
| STS-B-Vi | Qwen2-7B | TP | 70.33 |
| SICK-R-Vi | Qwen2-7B | TP | 69.59 |

## 8. PhoBERT Baseline

Cài thêm tokenizer tiếng Việt:

```bash
conda activate tp
pip install underthesea pyvi
```

Average pooling:

```bash
python evaluate_phobert_vi_sts.py \
  --model_name_or_path vinai/phobert-base-v2 \
  --task all \
  --encode_mode avg \
  --pooling mean \
  --word_segmenter underthesea \
  --batch_size 64 \
  --output_csv phobert_vi_results.csv
```

Prompt pooling tại token `<mask>`:

```bash
python evaluate_phobert_vi_sts.py \
  --model_name_or_path vinai/phobert-base-v2 \
  --task all \
  --encode_mode prompt \
  --word_segmenter underthesea \
  --batch_size 64 \
  --prompt_template 'This sentence: "*sent 0*" means <mask>' \
  --output_csv phobert_vi_results.csv
```

## Citation

```bibtex
@inproceedings{fu-etal-2025-token,
    title = "Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from {LLM}s",
    author = "Fu, Yuchen and Cheng, Zifeng and Jiang, Zhiwei and Wang, Zhonghui and Yin, Yafeng and Li, Zhengliang and Gu, Qing",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    pages = "3168--3181",
    url = "https://aclanthology.org/2025.acl-long.159/",
    doi = "10.18653/v1/2025.acl-long.159"
}
```
