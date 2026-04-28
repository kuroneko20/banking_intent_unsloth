import json
import os
import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Prompt CỰC NGẮN — ~50 tokens, không list labels
# Model học label name qua repetition (epochs x samples)
PROMPT_TEMPLATE = """### Instruction:
Classify the banking intent. Reply with ONLY the intent label, nothing else.

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
    # Shuffle để tránh model thấy cùng class liên tiếp
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    all_labels = sorted(train_df["intent"].unique().tolist())
    print(f"  -> {len(all_labels)} unique intents, {len(train_df)} samples")

    output_dir = config.get("output_dir", "outputs/banking-intent-model")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(all_labels, f, indent=2)
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
            formatted = PROMPT_TEMPLATE.format(text, intent) + tokenizer.eos_token
            texts.append(formatted)
        return {"formatted_text": texts}

    train_dataset = train_dataset.map(format_prompts, batched=True)

    # Verify token lengths
    lengths = [
        tokenizer(t, return_tensors="pt")["input_ids"].shape[1]
        for t in train_dataset["formatted_text"][:10]
    ]
    print(f"  -> Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
    if max(lengths) > config["max_seq_length"]:
        print(f"  ❌ TRUNCATION — tăng max_seq_length lên {max(lengths)+50}")
        return
    print(f"  ✅ OK, fit trong max_seq_length={config['max_seq_length']}")

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="formatted_text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=config.get("batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_ratio=0.05,
            num_train_epochs=config.get("num_train_epochs", 10),
            learning_rate=float(config.get("learning_rate", 1e-4)),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=config.get("optimizer", "adamw_8bit"),
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir="outputs",
            save_strategy="no",
        ),
    )

    print("Training... (loss nên giảm đều từ ~4.5 → ~0.3)")
    trainer.train()

    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    main()