import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================
# FIX 1: Prompt mới — liệt kê label hợp lệ ngay trong instruction
# Giúp model học được "danh sách nhãn" thay vì tự bịa
# ============================================================
def build_prompt_template(all_labels: list[str]) -> str:
    label_list_str = "\n".join(f"- {l}" for l in sorted(all_labels))
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "Classify the banking intent of the following input text.\n"
        "Output ONLY one exact intent label from the list below and nothing else. "
        "Do NOT add any explanation, punctuation, or extra words.\n\n"
        "Valid intent labels:\n"
        f"{label_list_str}\n\n"
        "### Input:\n"
        "{input}\n\n"
        "### Response:\n"
        "{response}"
    )


def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    print("Loading Data first to build label list...")
    train_df = pd.read_csv(config["train_data_path"])
    all_labels = sorted(train_df["intent"].unique().tolist())
    print(f"  -> Found {len(all_labels)} unique intents")

    prompt_template = build_prompt_template(all_labels)

    # Lưu label list ra file để inference dùng lại
    import json, os
    os.makedirs(config.get("output_dir", "outputs/banking-intent-model"), exist_ok=True)
    labels_path = os.path.join(config.get("output_dir", "outputs/banking-intent-model"), "labels.json")
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
        # FIX 2: Tăng r và lora_alpha để model có capacity học 77 classes
        r=config.get("lora_r", 32),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
            formatted = prompt_template.format(input=text, response=intent) + tokenizer.eos_token
            texts.append(formatted)
        return {"formatted_text": texts}

    train_dataset = train_dataset.map(format_prompts, batched=True)

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
            # FIX 3: Tăng epochs — 77 classes với 20 samples/class cần ít nhất 5 epochs
            num_train_epochs=config.get("num_train_epochs", 5),
            learning_rate=float(config.get("learning_rate", 2e-4)),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=config.get("optimizer", "adamw_8bit"),
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # FIX 4: cosine decay tốt hơn linear cho fine-tune
            seed=42,
            output_dir="outputs",
        ),
    )

    print("Training...")
    trainer.train()

    print(f"Saving model to {config['output_dir']}...")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print("Done!")


if __name__ == "__main__":
    main()