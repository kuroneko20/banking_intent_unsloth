import json
import os
import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================
# Prompt NGẮN GỌN — không nhét 77 labels vào
# max_seq_length=256 chỉ chứa được ~200 tokens prompt
# Toàn bộ sample (prompt + label) phải < 256 tokens
# ============================================================
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the banking intent of the following input text. Output ONLY the exact intent label and nothing else.

### Input:
{}

### Response:
{}"""


def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    print("Loading Data...")
    train_df = pd.read_csv(config["train_data_path"])
    all_labels = sorted(train_df["intent"].unique().tolist())
    print(f"  -> {len(all_labels)} unique intents, {len(train_df)} samples")

    # Lưu label list để inference dùng fuzzy match
    output_dir = config.get("output_dir", "outputs/banking-intent-model")
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"  -> Labels saved to {labels_path}")

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
            formatted = PROMPT_TEMPLATE.format(text, intent) + tokenizer.eos_token
            texts.append(formatted)
        return {"formatted_text": texts}

    train_dataset = train_dataset.map(format_prompts, batched=True)

    # Kiểm tra độ dài token thực tế của 5 mẫu đầu
    sample_lengths = []
    for t in train_dataset["formatted_text"][:5]:
        toks = tokenizer(t, return_tensors="pt")
        sample_lengths.append(toks["input_ids"].shape[1])
    print(f"  -> Sample token lengths (first 5): {sample_lengths}")
    print(f"  -> max_seq_length = {config['max_seq_length']}")
    if max(sample_lengths) > config["max_seq_length"]:
        print("  ⚠️  WARNING: Some samples exceed max_seq_length — sẽ bị truncate!")
    else:
        print("  ✅ Token lengths OK")

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="formatted_text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=config.get("batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
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