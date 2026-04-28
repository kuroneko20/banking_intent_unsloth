import json
import os
import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# FIX 1: Prompt rõ ràng hơn, có EOS token placeholder
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

    # FIX 2: Tăng lora_r lên 32 để có capacity tốt hơn cho 77 classes
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 32),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.get("lora_alpha", 64),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_dataset = Dataset.from_pandas(train_df)

    def format_prompts(examples):
        texts = []
        for text, intent in zip(examples["text"], examples["intent"]):
            # FIX 3: EOS token ở cuối để model học khi nào dừng
            formatted = PROMPT_TEMPLATE.format(text, intent) + tokenizer.eos_token
            texts.append(formatted)
        return {"formatted_text": texts}

    # FIX 4: Dùng include_groups=False để tránh DeprecationWarning (không liên quan train
    # nhưng preprocess đã được fix riêng)
    train_dataset = train_dataset.map(format_prompts, batched=True, num_proc=4)

    # Verify token lengths — check toàn bộ dataset thay vì chỉ 10 mẫu đầu
    print("  -> Verifying token lengths on full dataset...")
    sample_lengths = [
        len(tokenizer(t)["input_ids"])
        for t in train_dataset["formatted_text"]
    ]
    print(f"  -> Token lengths: min={min(sample_lengths)}, max={max(sample_lengths)}, avg={sum(sample_lengths)//len(sample_lengths)}")
    if max(sample_lengths) > config["max_seq_length"]:
        print(f"  ❌ TRUNCATION — tăng max_seq_length lên {max(sample_lengths)+50}")
        return
    print(f"  ✅ OK, fit trong max_seq_length={config['max_seq_length']}")

    # FIX 5: Tính warmup_steps thay vì warmup_ratio (deprecated trong v5.2)
    steps_per_epoch = len(train_dataset) // (
        config.get("batch_size", 4) * config.get("gradient_accumulation_steps", 8)
    )
    total_steps = steps_per_epoch * config.get("num_train_epochs", 15)
    warmup_steps = max(10, int(total_steps * 0.05))
    print(f"  -> Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="formatted_text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=4,
        args=TrainingArguments(
            per_device_train_batch_size=config.get("batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            # FIX 5: Dùng warmup_steps thay vì warmup_ratio
            warmup_steps=warmup_steps,
            # FIX 6: Tăng epochs để loss giảm xuống < 0.2
            num_train_epochs=config.get("num_train_epochs", 15),
            learning_rate=float(config.get("learning_rate", 2e-4)),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=config.get("optimizer", "adamw_8bit"),
            weight_decay=0.01,
            # FIX 7: cosine_with_restarts tốt hơn cho fine-tuning dài
            lr_scheduler_type="cosine_with_restarts",
            lr_scheduler_kwargs={"num_cycles": 3},
            seed=42,
            output_dir="outputs",
            save_strategy="no",
        ),
    )

    print("Training... (loss nên giảm từ ~4.5 → < 0.2)")
    trainer.train()

    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    main()