# Banking Intent Classification with Unsloth

Dự án Fine-tuning mô hình phân loại intent ngân hàng sử dụng dataset BANKING77 và Unsloth.

## 1. Setup Môi trường

Tạo virtual environment hoặc chạy trực tiếp trên Colab. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu

Script `preprocess_data.py` sẽ tải tập dữ liệu BANKING77, normalize text, và trích xuất ra một subset (20 sample/class cho train) để tối ưu thời gian fine-tune.
```bash
python scripts/preprocess_data.py
```

## 3. Huấn luyện (Training)

Quá trình training sử dụng mô hình Llama-3 (lượng tử hóa 4-bit) kết hợp với LoRA thông qua SFTTrainer.
```bash
bash train.sh
```

### Hoặc chạy trực tiếp:
```bash
python scripts/train.py
```

## 4. Suy luận (Inference)

Dự án cung cấp class `IntentClassification` nhận vào đường dẫn cấu hình để sinh dự đoán.
```bash
bash inference.sh
```

### Hoặc chạy trực tiếp:
```bash
python scripts/inference.py
```

## 5. Video Demo

(Sinh viên chèn link Google Drive công khai video demo quá trình inference, cho thấy input, output, và file code được thực thi tại đây).

👉 **[Link Video Demo Google Drive]**(Thay_bằng_link_của_bạn_tại_đây)