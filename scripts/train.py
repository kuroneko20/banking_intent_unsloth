import json
import os
import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback

# ------------------------------------------------------------------ #
# FIX 1: Prompt rõ ràng — liệt kê valid labels ngay trong instruction
# Model học predict đúng label string, không học "mô tả" intent.
# ------------------------------------------------------------------ #
PROMPT_TEMPLATE = """### Instruction:
You are a banking intent classifier. Given a customer message, respond with EXACTLY ONE label from the list below. Output only the label, nothing else.

### Valid labels:
{labels}

### Customer message:
{text}

### Label:
{response}"""


def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    print("Loading Data...")
    train_df = pd.read_csv(config["train_data_path"])
    train_df = train_df.dropna(subset=["text", "intent"])
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    all_labels = sorted(train_df["intent"].unique().tolist())
    labels_str = "\n".join(f"- {lb}" for lb in all_labels)
    print(f"  -> {len(all_labels)} unique intents, {len(train_df)} samples")

    output_dir = config.get("output_dir", "outputs/banking-intent-model")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"  -> Labels saved to {output_dir}/labels.json")

    print("Loading Model...")
    # FIX 2: Tăng max_seq_length lên 512 để không truncate prompt có label list
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config.get("max_seq_length", 512),
        dtype=None,
        load_in_4bit=config.get("load_in_4bit", True),
    )

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
            formatted = PROMPT_TEMPLATE.format(
                labels=labels_str,
                text=text,
                response=intent,  # Ground truth label — model học predict cái này
            ) + tokenizer.eos_token
            texts.append(formatted)
        return {"formatted_text": texts}

    train_dataset = train_dataset.map(format_prompts, batched=True, num_proc=2)

    # FIX 3: Verify token lengths — kiểm tra toàn bộ, dừng nếu bị truncate
    print("  -> Verifying token lengths...")
    sample_lengths = [
        len(tokenizer(t)["input_ids"])
        for t in train_dataset["formatted_text"]
    ]
    max_len = config.get("max_seq_length", 512)
    print(f"  -> Token lengths: min={min(sample_lengths)}, max={max(sample_lengths)}, "
          f"avg={sum(sample_lengths)//len(sample_lengths)}")
    truncated = sum(1 for l in sample_lengths if l >= max_len)
    if truncated > 0:
        print(f"  ⚠️  {truncated} samples hit max_seq_length={max_len} → tăng lên {max(sample_lengths)+50}")
        return
    print(f"  ✅ All samples fit within max_seq_length={max_len}")

    # FIX 4: Tính warmup_steps thay vì warmup_ratio (deprecated transformers v5+)
    batch_size = config.get("batch_size", 4)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    num_epochs = config.get("num_train_epochs", 5)  # FIX 5: giảm xuống 5 (không phải 15)

    steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(10, int(total_steps * 0.05))
    print(f"  -> Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}")

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="formatted_text",
        max_seq_length=max_len,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=float(config.get("learning_rate", 2e-4)),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=config.get("optimizer", "adamw_8bit"),
            weight_decay=0.01,
            # FIX 6: cosine schedule phù hợp hơn cho fine-tune ngắn
            lr_scheduler_type="cosine",
            seed=42,
            output_dir="outputs",
            save_strategy="no",  # Không lưu checkpoint giữa chừng để tiết kiệm thời gian
            report_to="none",    # Tắt wandb/tensorboard để không tốn overhead
        ),
    )

    print("Training... (loss mục tiêu: < 0.3)")
    trainer_output = trainer.train()

    final_loss = trainer_output.training_loss
    print(f"\n  -> Final training loss: {final_loss:.4f}")
    if final_loss > 0.5:
        print("  ⚠️  Loss > 0.5 — cân nhắc tăng epochs hoặc kiểm tra data quality")

    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()