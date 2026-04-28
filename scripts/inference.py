import json
import os
import yaml
import torch
import warnings
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# GIỐNG HỆT train.py — chỉ khác phần Response (không có label)
PROMPT_TEMPLATE = """### Instruction:
Classify the banking intent. Reply with ONLY the intent label, nothing else.

### Input:
{}

### Response:
"""


def normalize(text: str) -> str:
    """Chuẩn hóa text: lowercase, bỏ dấu câu thừa, thay space bằng underscore."""
    return text.strip().lower().strip(".,;:!?\"'() ").replace(" ", "_")


class IntentClassifier:
    def __init__(self, model_path: str):
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        self.checkpoint = config.get("model_checkpoint", "outputs/banking-intent-model")
        self.max_seq_length = config.get("max_seq_length", 256)

        labels_path = os.path.join(self.checkpoint, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.json not found in {self.checkpoint}")
        with open(labels_path) as f:
            self.valid_labels = json.load(f)
        print(f"[INFO] Loaded {len(self.valid_labels)} valid labels")

        # Pre-build lookup: normalized → original label
        self.label_norm_map = {normalize(lb): lb for lb in self.valid_labels}
        # Token set cho mỗi label (dùng cho fuzzy matching)
        self.label_tokens = {
            lb: set(lb.lower().split("_")) for lb in self.valid_labels
        }

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=config.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)

        # Stop tokens
        unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        candidates = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("</s>"),
        ]
        self.terminators = list({t for t in candidates if t is not None and t != unk_id})
        print(f"[INFO] Stop token IDs: {self.terminators}")

    def _find_best_label(self, raw: str) -> str:
        """
        4-bước matching (không dùng embedding — đó là nguyên nhân accuracy thấp):

        1. Exact match sau normalize
        2. Substring: label nằm trong output hoặc output nằm trong label
        3. Token overlap: label có nhiều token chung nhất với output
        4. SequenceMatcher (edit distance) — fallback cuối
        """
        # Lấy dòng đầu tiên, normalize
        first_line = raw.strip().splitlines()[0]
        cleaned = normalize(first_line)

        # Bước 1: Exact match
        if cleaned in self.label_norm_map:
            return self.label_norm_map[cleaned]

        # Thử thêm: exact match không có underscore
        cleaned_nospace = cleaned.replace("_", "").replace(" ", "")
        for norm, orig in self.label_norm_map.items():
            if norm.replace("_", "") == cleaned_nospace:
                return orig

        # Bước 2: Substring match
        for norm, orig in self.label_norm_map.items():
            if norm in cleaned or cleaned in norm:
                return orig

        # Bước 3: Token overlap (Jaccard-like)
        output_tokens = set(cleaned.replace("_", " ").split())
        best_overlap = -1
        best_label_overlap = None
        for lb, lb_tokens in self.label_tokens.items():
            if not lb_tokens:
                continue
            overlap = len(output_tokens & lb_tokens) / len(output_tokens | lb_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label_overlap = lb

        # Nếu overlap tốt (≥ 0.4), dùng ngay
        if best_overlap >= 0.4:
            return best_label_overlap

        # Bước 4: SequenceMatcher (edit distance proxy)
        best_ratio = -1.0
        best_label_fuzzy = None
        for norm, orig in self.label_norm_map.items():
            ratio = SequenceMatcher(None, cleaned, norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_label_fuzzy = orig

        return best_label_fuzzy

    def __call__(self, message: str) -> str:
        prompt = PROMPT_TEMPLATE.format(message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,       # Tăng lên 20 để cover label dài
                use_cache=True,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,         # Không dùng sampling
                repetition_penalty=1.1,  # Giảm xuống 1.1 — 1.2 có thể block label hợp lệ
            )

        new_tokens = outputs[0][input_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predicted = self._find_best_label(raw)

        del inputs, outputs
        torch.cuda.empty_cache()
        return predicted


if __name__ == "__main__":
    print("Initializing Model...")
    classifier = IntentClassifier(model_path="configs/inference.yaml")

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

        # Thêm: In các label hay bị sai để debug
        wrong = [(y_true_clean[i], y_pred_clean[i]) for i in range(total) if y_true_clean[i] != y_pred_clean[i]]
        if wrong:
            from collections import Counter
            print("\n--- TOP 10 MOST CONFUSED PAIRS ---")
            for (true, pred), count in Counter(wrong).most_common(10):
                print(f"  TRUE: '{true}' → PRED: '{pred}' ({count}x)")

    except FileNotFoundError:
        print("Error: sample_data/test.csv not found.")