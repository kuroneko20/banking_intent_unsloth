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


def build_prompt(label_list_str: str, input_text: str) -> str:
    """Giống hệt train.py, chỉ bỏ phần response."""
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
    )


def fuzzy_match_label(raw_pred: str, valid_labels: list) -> str:
    first_line = raw_pred.strip().splitlines()[0].strip().lower()

    # 1. Exact match
    for label in valid_labels:
        if label.lower() == first_line:
            return label

    # 2. Label nằm trong output (bỏ dấu câu thừa)
    cleaned = first_line.strip(".,;:!?\"'()")
    for label in valid_labels:
        if label.lower() == cleaned:
            return label

    # 3. Substring match
    for label in valid_labels:
        if label.lower() in first_line:
            return label

    # 4. Fuzzy difflib
    matches = get_close_matches(first_line, [l.lower() for l in valid_labels], n=1, cutoff=0.55)
    if matches:
        for label in valid_labels:
            if label.lower() == matches[0]:
                return label

    return first_line


class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        self.checkpoint = config.get("model_checkpoint", "outputs/banking-intent-model")
        self.max_seq_length = config.get("max_seq_length", 1024)

        # Load labels và label_list_str từ train
        labels_path = os.path.join(self.checkpoint, "labels.json")
        label_str_path = os.path.join(self.checkpoint, "label_list_str.txt")

        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self.valid_labels = json.load(f)
            print(f"[INFO] Loaded {len(self.valid_labels)} valid labels")
        else:
            raise FileNotFoundError(f"labels.json not found in {self.checkpoint}. Retrain first.")

        if os.path.exists(label_str_path):
            with open(label_str_path) as f:
                self.label_list_str = f.read().strip()
        else:
            # Fallback: tự build lại
            self.label_list_str = ", ".join(sorted(self.valid_labels))

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=config.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)

        unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        candidates = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("</s>"),
        ]
        self.terminators = list({t for t in candidates if t is not None and t != unk_id})
        print(f"[INFO] Stop token IDs: {self.terminators}")

    def __call__(self, message: str) -> str:
        prompt = build_prompt(self.label_list_str, message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                use_cache=True,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.2,
            )

        new_tokens = outputs[0][input_len:]
        raw_label = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predicted = fuzzy_match_label(raw_label, self.valid_labels)

        del inputs, outputs
        torch.cuda.empty_cache()

        return predicted


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
    try:
        df_test = pd.read_csv("sample_data/test.csv")
        y_true = df_test["intent"].tolist()
        texts = df_test["text"].tolist()
        y_pred = []
        total = len(texts)
        print(f"Predicting {total} samples...")

        for i, text in enumerate(texts):
            pred = classifier(message=text)
            y_pred.append(pred)
            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(f"  -> Processed {i+1}/{total} | last pred: '{pred}'")

        y_true_clean = [str(y).strip().lower() for y in y_true]
        y_pred_clean = [str(y).strip().lower() for y in y_pred]
        acc = accuracy_score(y_true_clean, y_pred_clean)

        print("\n==========================================")
        print(f"✅ FINAL ACCURACY ON TEST SET: {acc * 100:.2f}%")
        print("==========================================")

        print("\n--- DEBUG: First 10 predictions ---")
        for i in range(min(10, total)):
            mark = "✅" if y_true_clean[i] == y_pred_clean[i] else "❌"
            print(f"{mark} TRUE: '{y_true_clean[i]}' | PRED: '{y_pred_clean[i]}'")

    except FileNotFoundError:
        print("Error: sample_data/test.csv not found. Run preprocess_data.py first.")