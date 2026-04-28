import os
import pandas as pd
from datasets import load_dataset


def preprocess():
    print("Loading BANKING77 dataset...")
    dataset = load_dataset("mteb/banking77")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    # Chuẩn hóa cột intent
    if "label_text" in df_train.columns:
        df_train["intent"] = df_train["label_text"]
        df_test["intent"] = df_test["label_text"]
    else:
        df_train["intent"] = df_train["label"]
        df_test["intent"] = df_test["label"]

    df_train["text"] = df_train["text"].str.lower().str.strip()
    df_test["text"] = df_test["text"].str.lower().str.strip()

    # FIX: Tăng TRAIN_N lên 77 (dùng toàn bộ data gốc ~770 samples/class nếu đủ)
    # BANKING77 có ~100 train samples/class → lấy tất cả để tối đa signal
    TRAIN_N = 70   # lấy hết (dataset gốc có 100 samples/class)
    TEST_N = 5

    df_train_sampled = (
        df_train.groupby("intent", group_keys=False)[["text", "intent"]]
        .apply(lambda g: g.sample(n=min(TRAIN_N, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    df_test_sampled = (
        df_test.groupby("intent", group_keys=False)[["text", "intent"]]
        .apply(lambda g: g.sample(n=min(TEST_N, len(g)), random_state=42))
        .reset_index(drop=True)
    )

    os.makedirs("sample_data", exist_ok=True)
    df_train_sampled.to_csv("sample_data/train.csv", index=False)
    df_test_sampled.to_csv("sample_data/test.csv", index=False)

    n_classes = df_train_sampled["intent"].nunique()
    sizes = df_train_sampled.groupby("intent").size()
    print(f"Train: {len(df_train_sampled)} samples | {n_classes} classes")
    print(f"  samples/class: min={sizes.min()}, max={sizes.max()}, avg={sizes.mean():.1f}")
    print(f"Test:  {len(df_test_sampled)} samples | {df_test_sampled['intent'].nunique()} classes")
    print("Saved to sample_data/")


if __name__ == "__main__":
    preprocess()