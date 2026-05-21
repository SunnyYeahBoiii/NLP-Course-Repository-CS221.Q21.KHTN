# Chạy code Token Prepending hiện tại

Tài liệu này mô tả cách chạy code trong thư mục `token_prepending` của repo hiện tại. Các lệnh giả định bạn đang đứng ở root repo `NLP-Course-Repository-CS221.Q21.KHTN`.

## Tổng quan runtime

- Entry point chính: `token_prepending/evaluate.py`
- Script chạy nhanh: `token_prepending/run.sh`
- Config mặc định: `token_prepending/config.yaml`
- Model mặc định hiện tại: `Qwen/Qwen2.5-7B`
- Task mặc định hiện tại: `vi-sts`
- Cache Hugging Face mặc định: `token_prepending/.cache/huggingface`
- Notebook Kaggle: `token_prepending/token_prepending_kaggle_t4x2.ipynb`

Lưu ý quan trọng: `run.sh`, `config.yaml`, và đường dẫn `SentEval` dùng đường dẫn tương đối theo thư mục `token_prepending`. Vì vậy nên `cd token_prepending` trước khi chạy.

## 1. Tạo môi trường Python

Khuyến nghị dùng Python 3.10 để tránh xung đột với các dependency đang pin version cũ.

```bash
cd token_prepending
conda create -n tp python=3.10 -y
conda activate tp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu cài dependency trên macOS bị lỗi vì package CUDA-only như `bitsandbytes`, dùng Kaggle theo mục 5 hoặc cài runtime tối thiểu theo notebook Kaggle.

## 2. Chạy mặc định

```bash
cd token_prepending
bash run.sh
```

Lệnh tương đương nếu muốn chạy từ root repo:

```bash
(cd token_prepending && bash run.sh)
```

Không chạy trực tiếp `bash token_prepending/run.sh` từ root repo, vì script sẽ không tìm đúng `config.yaml` và `SentEval`.

## 3. Chạy bằng `evaluate.py`

```bash
cd token_prepending
python evaluate.py \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --task_set vi-sts \
  --use_which_plan tp \
  --device auto \
  --cache_dir .cache/huggingface \
  --vietnamese_dataset_name nemixo/stsbenchmark-sts-vietnamese \
  --vietnamese_split test
```

Ý nghĩa chính:

- `--task_set vi-sts`: chạy Vietnamese STS từ Hugging Face, không cần tải SentEval English data.
- `--use_which_plan tp`: bật Token Prepending.
- `--device auto`: ưu tiên MPS trên Apple Silicon, sau đó CUDA, cuối cùng CPU.
- `--cache_dir .cache/huggingface`: lưu model/dataset trong thư mục project.

## 4. Chạy test nhỏ

Sau khi cài dependency:

```bash
cd token_prepending
python -m unittest discover tests
```

Nếu gặp lỗi:

```text
ModuleNotFoundError: No module named 'torch'
```

nghĩa là chưa active đúng environment hoặc chưa cài `requirements.txt`.

## 5. Chạy trên Kaggle T4 x2

1. Mở hoặc upload `token_prepending/token_prepending_kaggle_t4x2.ipynb` lên Kaggle.
2. Trong notebook settings, chọn `Accelerator -> GPU T4 x2`.
3. Bật Internet nếu cần tải model/dataset từ Hugging Face.
4. Chạy toàn bộ notebook.

Notebook sẽ tự:

- cài minimal runtime phù hợp Kaggle;
- kiểm tra có 2 CUDA GPU;
- ghi source vào `/kaggle/working/token_prepending`;
- chạy Vietnamese STS mặc định.

Không bật `--tensor_parallel` cho notebook hiện tại, vì nhánh đó dùng `AutoModelForCausalLM` và bypass custom TP model classes trong `senllm`.

## 6. Chạy English SentEval

Các task `sts`, `stsb`, `transfer`, `full` cần dữ liệu SentEval. Tải dữ liệu trước:

```bash
cd token_prepending/SentEval/data/downstream
bash download_dataset.sh
cd ../../../
```

Sau đó chạy config English, ví dụ:

```bash
bash run.sh llama-2-7b-tp config.yaml
```

Các model Llama có thể cần quyền truy cập Hugging Face hoặc local path hợp lệ trong `config.yaml`.

## 7. Lỗi thường gặp

- `config file config.yaml not found`: chạy từ `token_prepending`, hoặc dùng `(cd token_prepending && bash run.sh)`.
- `ModuleNotFoundError: torch`: active đúng env `tp` và cài dependency.
- `CUDA out of memory`: giảm `batch_size` xuống `1`, dùng `device: auto`, hoặc chạy Kaggle T4 x2.
- `SentEval/data` missing: chỉ ảnh hưởng English SentEval; dùng `task_set: vi-sts` nếu chưa tải dữ liệu English.
- Tải model lâu hoặc hết disk: xóa cache tại `token_prepending/.cache/huggingface` nếu muốn tải lại từ đầu.

## 8. Kiểm tra nhanh trước khi chạy full model

```bash
cd token_prepending
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mps:", torch.backends.mps.is_available())
PY
```

Nếu cả CUDA và MPS đều `False`, vẫn có thể chạy CPU nhưng model 7B sẽ rất chậm và dễ thiếu RAM.
