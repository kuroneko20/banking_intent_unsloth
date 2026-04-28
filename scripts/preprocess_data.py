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

    print("Sampling subset...")
    TRAIN_N = 50
    TEST_N = 5

    # FIX: Dùng include_groups=False để tránh DeprecationWarning trong pandas >= 2.2
    df_train_sampled = (
        df_train.groupby("intent", group_keys=False)
        .apply(lambda g: g.sample(n=min(TRAIN_N, len(g)), random_state=42), include_groups=False)
        .reset_index(drop=True)
    )
    df_test_sampled = (
        df_test.groupby("intent", group_keys=False)
        .apply(lambda g: g.sample(n=min(TEST_N, len(g)), random_state=42), include_groups=False)
        .reset_index(drop=True)
    )

    # include_groups=False loại bỏ cột groupby khỏi kết quả — cần add lại
    # Cách an toàn hơn: merge lại với index gốc
    # Thực ra với include_groups=False, cột 'intent' sẽ bị drop trong lambda result
    # nên dùng cách khác:
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

    print(f"Train samples: {len(df_train_sampled)} Samples/class: {df_train_sampled.groupby('intent').size().min()}")
    print(f"Train: {len(df_train_sampled)} samples across {df_train_sampled['intent'].nunique()} classes")
    print(f"Test:  {len(df_test_sampled)} samples across {df_test_sampled['intent'].nunique()} classes")
    print(f"Train samples/class: min={df_train_sampled.groupby('intent').size().min()}, max={df_train_sampled.groupby('intent').size().max()}")
    print("Saved to sample_data/")

if __name__ == "__main__":
    preprocess()