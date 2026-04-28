import json
import os
import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments


def build_prompt(label_list_str: str, input_text: str, response: str = "") -> str:
    """
    Prompt nhét label list dạng comma-separated (ngắn hơn bullet list ~40%).
    Tổng ~700 tokens — an toàn với max_seq_length=1024.
    """
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "Classify the banking intent of the following input text.\n"
        "You MUST output ONLY one label from this exact list, word-for-word, nothing else:\n"
        f"{label_list_str}\n\n"
        "### Input:\n"
        f"{input_text}\n\n"
        "### Response:\n"
        f"{response}"
    )


def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Bắt buộc max_seq_length >= 1024 để chứa prompt + label
    if config.get("max_seq_length", 256) < 1024:
        print("⚠️  WARNING: max_seq_length < 1024 — label list sẽ bị truncate!")
        print("   Hãy set max_seq_length: 1024 trong configs/train.yaml")

    print("Loading Data...")
    train_df = pd.read_csv(config["train_data_path"])
    all_labels = sorted(train_df["intent"].unique().tolist())
    label_list_str = ", ".join(all_labels)  # comma-separated, ngắn gọn
    print(f"  -> {len(all_labels)} unique intents, {len(train_df)} samples")

    # Lưu labels + label_list_str để inference dùng lại
    output_dir = config.get("output_dir", "outputs/banking-intent-model")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(all_labels, f, indent=2)
    with open(os.path.join(output_dir, "label_list_str.txt"), "w") as f:
        f.write(label_list_str)
    print(f"  -> Labels saved to {output_dir}/labels.json")

    print("Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_dataset = Dataset.from_pandas(train_df)

    def format_prompts(examples):
        texts = []
        for text, intent in zip(examples["text"], examples["intent"]):
            formatted = build_prompt(label_list_str, text, intent) + tokenizer.eos_token
            texts.append(formatted)
        return {"formatted_text": texts}

    train_dataset = train_dataset.map(format_prompts, batched=True)

    # Kiểm tra token length thực tế
    sample_lengths = [
        tokenizer(t, return_tensors="pt")["input_ids"].shape[1]
        for t in train_dataset["formatted_text"][:5]
    ]
    print(f"  -> Token lengths (5 samples): {sample_lengths}")
    if max(sample_lengths) > config["max_seq_length"]:
        print(f"  ❌ TRUNCATION DETECTED! Max sample={max(sample_lengths)} > max_seq_length={config['max_seq_length']}")
        print("     Set max_seq_length: 1024 in train.yaml and retrain!")
        return
    else:
        print(f"  ✅ All samples fit within max_seq_length={config['max_seq_length']}")

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="formatted_text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=config.get("batch_size", 2),  # batch kéo nhỏ vì seq dài hơn
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            warmup_steps=10,
            num_train_epochs=config.get("num_train_epochs", 5),
            learning_rate=float(config.get("learning_rate", 2e-4)),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=config.get("optimizer", "adamw_8bit"),
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir="outputs",
        ),
    )

    print("Training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()