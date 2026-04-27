import os
import pandas as pd
from datasets import load_dataset

def preprocess():
    print("Loading BANKING77 dataset...")
    # Tải dataset mteb/banking77
    dataset = load_dataset("mteb/banking77")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    
    # Dataset mteb/banking77 đã có sẵn cột 'label_text' chứa tên intent dạng text
    # Nên ta gán thẳng luôn sang cột 'intent':
    if "label_text" in df_train.columns:
        df_train["intent"] = df_train["label_text"]
        df_test["intent"] = df_test["label_text"]
    else:
        # Dự phòng trường hợp cột mang tên khác
        df_train["intent"] = df_train["label"]
        df_test["intent"] = df_test["label"]
    
    # Text Normalization: Đưa về chữ thường và xóa khoảng trắng thừa
    df_train["text"] = df_train["text"].str.lower().str.strip()
    df_test["text"] = df_test["text"].str.lower().str.strip()
    
    # Sampling: Chọn 20 mẫu/class cho train và 5 mẫu/class cho test để chạy nhanh
    print("Sampling subset for faster training...")
    df_train_sampled = df_train.groupby("intent").sample(n=20, random_state=42)
    df_test_sampled = df_test.groupby("intent").sample(n=5, random_state=42)
    
    # Save to CSV
    os.makedirs("sample_data", exist_ok=True)
    df_train_sampled.to_csv("sample_data/train.csv", index=False)
    df_test_sampled.to_csv("sample_data/test.csv", index=False)
    print("Data saved to sample_data/train.csv and sample_data/test.csv")

if __name__ == "__main__":
    preprocess()