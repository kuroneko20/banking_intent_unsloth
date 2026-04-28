import os
import pandas as pd
from datasets import load_dataset

def preprocess():
    print("Loading BANKING77 dataset...")
    dataset = load_dataset("mteb/banking77")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    if "label_text" in df_train.columns:
        df_train["intent"] = df_train["label_text"]
        df_test["intent"] = df_test["label_text"]
    else:
        df_train["intent"] = df_train["label"]
        df_test["intent"] = df_test["label"]

    df_train["text"] = df_train["text"].str.lower().str.strip()
    df_test["text"] = df_test["text"].str.lower().str.strip()

    # Tăng từ 20 → 50 samples/class cho train (~2.5x data)
    # banking77 train có ~130 samples/class nên 50 vẫn đủ
    print("Sampling subset...")
    df_train_sampled = df_train.groupby("intent").sample(n=50, random_state=42)

    df_test_sampled = df_test.groupby("intent").sample(n=5, random_state=42)

    os.makedirs("sample_data", exist_ok=True)
    df_train_sampled.to_csv("sample_data/train.csv", index=False)
    df_test_sampled.to_csv("sample_data/test.csv", index=False)

    print(f"Train: {len(df_train_sampled)} samples ({len(df_train_sampled)//77}/class)")
    print(f"Test:  {len(df_test_sampled)} samples ({len(df_test_sampled)//77}/class)")
    print("Saved to sample_data/")

if __name__ == "__main__":
    preprocess()