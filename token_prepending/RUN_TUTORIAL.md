# Tutorial chạy code Token Prepending

File này hướng dẫn chạy nhánh code hiện tại trong `token_prepending`. Các lệnh giả định bạn đang đứng ở root repo `NLP-Course-Repository-CS221.Q21.KHTN`.

## 1. Chạy local mặc định

Mặc định hiện tại chạy `Qwen/Qwen2.5-7B`, task `vi-sts`, TP mode, cache model/dataset vào `token_prepending/.cache/huggingface`.

```bash
cd token_prepending
conda create -n tp python=3.10 -y
conda activate tp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash run.sh
```

Lệnh tương đương nếu đang đứng ở root repo:

```bash
(cd token_prepending && bash run.sh)
```

Không chạy `bash token_prepending/run.sh` từ root repo, vì `run.sh`, `config.yaml`, và đường dẫn `SentEval` hiện đang dùng đường dẫn tương đối theo thư mục `token_prepending`.

## 2. Chạy bằng lệnh Python trực tiếp

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

Nếu máy không đủ RAM/VRAM cho model 7B, dùng Kaggle T4 x2 theo mục 4.

## 3. Chạy test nhỏ

Sau khi cài dependency:

```bash
cd token_prepending
python -m unittest discover tests
```

Nếu gặp `ModuleNotFoundError: No module named 'torch'`, nghĩa là chưa active đúng environment hoặc chưa cài dependency.

## 4. Chạy trên Kaggle T4 x2

1. Upload hoặc mở `token_prepending/token_prepending_kaggle_t4x2.ipynb` trên Kaggle.
2. Vào notebook settings, chọn `Accelerator -> GPU T4 x2`.
3. Bật Internet nếu cần tải model/dataset từ Hugging Face.
4. Chạy toàn bộ notebook.

Notebook tự cài minimal runtime, kiểm tra có 2 CUDA GPU, ghi source vào `/kaggle/working/token_prepending`, rồi chạy Vietnamese STS mặc định. Notebook không dùng `--tensor_parallel`, vì đường đó bypass custom TP model classes.

## 5. Chạy English SentEval

Các task `sts`, `stsb`, `transfer`, `full` cần dữ liệu SentEval. Tải trước:

```bash
cd token_prepending/SentEval/data/downstream
bash download_dataset.sh
cd ../../../
```

Sau đó chạy ví dụ:

```bash
bash run.sh llama-2-7b-tp config.yaml
```

Các model Llama có thể cần quyền truy cập Hugging Face và local path đúng trong `config.yaml`.

## 6. Lỗi thường gặp

- `config file config.yaml not found`: chạy lệnh từ `token_prepending`, hoặc dùng `(cd token_prepending && bash run.sh)`.
- `ModuleNotFoundError: torch`: cài dependency trong environment Python 3.10.
- `CUDA out of memory`: giảm `batch_size` xuống `1`, dùng `device: auto`, hoặc chạy notebook Kaggle T4 x2.
- `SentEval/data` missing: chỉ ảnh hưởng English SentEval; dùng `task_set: vi-sts` nếu chưa tải dữ liệu English.
- Tải model lâu hoặc hết disk: xóa cache ở `token_prepending/.cache/huggingface` khi cần chạy lại từ đầu.
