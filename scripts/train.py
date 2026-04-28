"""
SetFit trainer cho Banking77 intent classification.
SetFit = Sentence Transformer fine-tuning với contrastive learning.
Không cần LLM, không cần GPU lớn, accuracy ~85% với 50 samples/class.
"""
import json
import os
import yaml
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder


def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    output_dir = config.get("output_dir", "outputs/banking-intent-model")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Data...")
    train_df = pd.read_csv(config["train_data_path"])
    print(f"  -> {train_df['intent'].nunique()} classes, {len(train_df)} samples")

    # Encode labels thành integer (SetFit yêu cầu)
    all_labels = sorted(train_df["intent"].unique().tolist())
    le = LabelEncoder()
    le.fit(all_labels)
    train_df["label"] = le.transform(train_df["intent"])

    # Lưu label mapping để inference decode về tên
    label_mapping = {int(i): name for i, name in enumerate(le.classes_)}
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"  -> {len(all_labels)} labels saved to {output_dir}/labels.json")

    train_dataset = Dataset.from_dict({
        "text": train_df["text"].tolist(),
        "label": train_df["label"].tolist(),
    })

    print("Loading SetFit Model...")
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        labels=all_labels,
    )

    print("Initializing Trainer...")
    args = TrainingArguments(
        output_dir=output_dir,
        batch_size=32,
        num_epochs=1,               # SetFit chỉ cần 1 epoch contrastive learning
        num_iterations=20,          # số cặp sentence tạo ra mỗi class
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    print("Training... (kỳ vọng 3-8 phút trên T4)")
    trainer.train()

    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()