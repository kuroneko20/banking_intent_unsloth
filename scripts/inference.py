import json
import os
import yaml
import torch
import warnings
import pandas as pd
from difflib import get_close_matches
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def build_prompt_template(all_labels: list[str]) -> str:
    """Phải giống hệt 100% với prompt trong train.py"""
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
        "{{input}}\n\n"
        "### Response:\n"
    )


def fuzzy_match_label(raw_pred: str, valid_labels: list[str]) -> str:
    """
    Nếu model output không khớp chính xác với label hợp lệ,
    tìm label gần nhất thay vì trả về text rác.
    Ưu tiên: exact match → substring match → difflib fuzzy match
    """
    raw_lower = raw_pred.strip().lower()

    # 1. Exact match (case-insensitive)
    for label in valid_labels:
        if label.lower() == raw_lower:
            return label

    # 2. Lấy dòng đầu tiên nếu model vẫn babble nhiều dòng
    first_line = raw_lower.splitlines()[0].strip()
    for label in valid_labels:
        if label.lower() == first_line:
            return label

    # 3. Substring match
    for label in valid_labels:
        if label.lower() in first_line or first_line in label.lower():
            return label

    # 4. Fuzzy match với difflib
    matches = get_close_matches(first_line, [l.lower() for l in valid_labels], n=1, cutoff=0.6)
    if matches:
        for label in valid_labels:
            if label.lower() == matches[0]:
                return label

    # 5. Không match được gì → trả về dòng đầu để debug
    return raw_pred.strip().splitlines()[0].strip()


class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        self.checkpoint = config.get("model_checkpoint", "outputs/banking-intent-model")
        self.max_seq_length = config.get("max_seq_length", 256)

        # Load danh sách label hợp lệ được lưu lúc train
        labels_path = os.path.join(self.checkpoint, "labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.valid_labels = json.load(f)
            print(f"[INFO] Loaded {len(self.valid_labels)} valid labels from {labels_path}")
        else:
            self.valid_labels = None
            print("[WARN] labels.json not found — fuzzy matching disabled. Run train.py first.")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=config.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)

        # Build prompt (phải khớp hệt train)
        if self.valid_labels:
            pt = build_prompt_template(self.valid_labels)
            self.prompt_template = pt.replace("{{input}}", "{}")
        else:
            self.prompt_template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                "Classify the banking intent of the following input text. "
                "Output ONLY the exact intent label and nothing else.\n\n"
                "### Input:\n{}\n\n### Response:\n"
            )

        # Stop tokens an toàn
        candidate_stop_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("</s>"),
        ]
        unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        self.terminators = list({
            t for t in candidate_stop_tokens
            if t is not None and t != unk_id
        })
        print(f"[INFO] Stop token IDs: {self.terminators}")

    def __call__(self, message: str) -> str:
        prompt = self.prompt_template.format(message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                use_cache=True,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.1,
            )

        new_tokens = outputs[0][input_len:]
        raw_label = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if self.valid_labels:
            predicted_label = fuzzy_match_label(raw_label, self.valid_labels)
        else:
            predicted_label = raw_label.splitlines()[0].strip()

        del inputs, outputs
        torch.cuda.empty_cache()

        return predicted_label


if __name__ == "__main__":
    print("Initializing Model...")
    classifier = IntentClassification(model_path="configs/inference.yaml")

    test_messages = [
        "I lost my card yesterday, please help me block it.",
        "What is the exchange rate for USD to EUR?",
        "Why was I charged an extra fee for my ATM withdrawal?"
    ]

    print("\n--- 1. INFERENCE DEMO ---")
    for msg in test_messages:
        intent = classifier(message=msg)
        print(f"Input:            {msg}")
        print(f"Predicted Intent: {intent}\n")

    print("\n--- 2. EVALUATING ON TEST SET ---")
    print("Loading sample_data/test.csv...")
    try:
        df_test = pd.read_csv("sample_data/test.csv")
        y_true = df_test["intent"].tolist()
        texts = df_test["text"].tolist()
        y_pred = []

        total_samples = len(texts)
        print(f"Predicting {total_samples} samples...")

        for i, text in enumerate(texts):
            pred = classifier(message=text)
            y_pred.append(pred)
            if (i + 1) % 20 == 0 or (i + 1) == total_samples:
                print(f"  -> Processed {i + 1}/{total_samples} | last pred: '{pred}'")

        y_true_clean = [str(y).strip().lower() for y in y_true]
        y_pred_clean = [str(y).strip().lower() for y in y_pred]

        acc = accuracy_score(y_true_clean, y_pred_clean)

        print("\n==========================================")
        print(f"✅ FINAL ACCURACY ON TEST SET: {acc * 100:.2f}%")
        print("==========================================")

        print("\n--- DEBUG: First 10 predictions vs ground truth ---")
        for i in range(min(10, total_samples)):
            match = "✅" if y_true_clean[i] == y_pred_clean[i] else "❌"
            print(f"{match} TRUE: '{y_true_clean[i]}' | PRED: '{y_pred_clean[i]}'")

    except FileNotFoundError:
        print("Error: Could not find sample_data/test.csv. Please run preprocess_data.py first.")