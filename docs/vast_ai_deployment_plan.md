# Kế hoạch triển khai Token Prepending trên Vast.ai

## 1. Yêu cầu tài nguyên (Khuyến nghị trên Vast.ai)
- **GPU:** 1x GPU có tối thiểu **24GB VRAM** (ví dụ: RTX 3090, RTX 4090, RTX A5000, hoặc A100).
- **Image:** Chọn các template có sẵn của **PyTorch** (ví dụ: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` hoặc các template deep learning mặc định của Vast.ai).
- **Disk Space:** Khuyến nghị cấp phát tối thiểu **50GB** (để chứa OS, dataset SentEval, và model weights LLaMA-2-7B tải từ Hugging Face).

## 2. Các bước thực hiện chi tiết trên Terminal của Server

### Bước 1: Clone Repository
```bash
git clone https://github.com/fuyuchenIfyw/token_prepending.git
cd token_prepending
```

### Bước 2: Xử lý file requirements.txt
Bởi vì Image Vast.ai đã cài sẵn PyTorch tối ưu cho CUDA của máy đó, việc ép cài `torch==2.5.1` có thể gây gỡ cài đặt phiên bản cũ và xung đột CUDA.
```bash
# Xóa dòng torch==2.5.1 khỏi requirements.txt để dùng PyTorch có sẵn của Image
sed -i '/torch==2.5.1/d' requirements.txt
```

### Bước 3: Thiết lập môi trường ảo và cài đặt thư viện
```bash
# Tạo môi trường ảo (venv) để quản lý package độc lập
python3 -m venv venv
source venv/bin/activate

# Cài đặt các dependencies
pip install -r requirements.txt
```

### Bước 4: Tải dữ liệu SentEval
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
cd ../../../
```

### Bước 5: Xác thực Hugging Face (Yêu cầu bắt buộc)
Bạn cần có tài khoản Hugging Face và đã xin quyền truy cập (Accept License) model `meta-llama/Llama-2-7b-hf`.
```bash
huggingface-cli login
# Dán token của bạn vào (lấy tại https://huggingface.co/settings/tokens)
```

### Bước 6: Cập nhật file `config.yaml`
Cập nhật đường dẫn `model_name_or_path` trong file `config.yaml` từ đường dẫn local của tác giả thành repository trên HuggingFace.
```bash
# Cập nhật đường dẫn local thành "meta-llama/Llama-2-7b-hf"
sed -i 's|/tos-bjml-ai4chem/fuyuchen/llm_weights/Llama-2-7b-hf|meta-llama/Llama-2-7b-hf|g' config.yaml
sed -i 's|/path/to/Llama-2-7b-hf|meta-llama/Llama-2-7b-hf|g' config.yaml
```
*Mẹo: Nếu GPU của bạn có VRAM ít hơn 24GB và gặp lỗi Out Of Memory (OOM), hãy dùng nano hoặc vim để mở `config.yaml` và giảm `batch_size` từ 16 xuống 8 hoặc 4.*

### Bước 7: Thực thi chạy mô hình
Chạy lệnh bash để bắt đầu đánh giá với phương pháp Token Prepending:
```bash
bash run.sh llama-2-7b-tp config.yaml
```
