import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Sử dụng Format Alpaca cực mạnh để ép mô hình tuân thủ tuyệt đối
prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

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
    
    print("Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print("Loading Data...")
    train_df = pd.read_csv(config["train_data_path"])
    train_dataset = Dataset.from_pandas(train_df)
    
    def format_prompts(examples):
        texts = []
        for text, intent in zip(examples["text"], examples["intent"]):
            # Dạy mô hình điểm dừng tuyệt đối
            formatted_text = prompt_template.format(text, intent) + tokenizer.eos_token
            texts.append(formatted_text)
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
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_steps=5,
            num_train_epochs=config["num_train_epochs"], # Vẫn giữ là 3 epochs nhé
            learning_rate=float(config["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=config["optimizer"],
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
        ),
    )
    
    print("Training...")
    trainer.train()
    
    print(f"Saving model to {config['output_dir']}...")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()