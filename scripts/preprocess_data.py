import os
import pandas as pd
from datasets import load_dataset

def preprocess():
    print("Loading BANKING77 dataset...")
    # Đã thêm trust_remote_code=True để bypass lỗi bảo mật của Hugging Face
    dataset = load_dataset("mteb/banking77")
    labels = dataset["train"].features["label"].names
    
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    
    # Label Mapping: Chuyển ID thành tên Intent dạng text
    df_train["intent"] = df_train["label"].apply(lambda x: labels[x])
    df_test["intent"] = df_test["label"].apply(lambda x: labels[x])
    
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