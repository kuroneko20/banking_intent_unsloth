"""
inference.py — Banking Intent Classifier
========================================
FIX CHÍNH: Thay toàn bộ fuzzy matching bằng log-prob scoring.

Cách hoạt động:
- Với mỗi input, build prompt (không có response)
- Với mỗi valid label, nối prompt + label và tính log-prob của phần label
- Pick label có log-prob cao nhất → guaranteed trả về valid label, không cần fuzzy match
- Hiệu quả hơn vì sử dụng đúng signal từ model (không phải string similarity)
"""

import json
import os
import yaml
import torch
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# PHẢI GIỐNG HỆT train.py (không có response)
PROMPT_TEMPLATE = """### Instruction:
You are a banking intent classifier. Given a customer message, respond with EXACTLY ONE label from the list below. Output only the label, nothing else.

### Valid labels:
{labels}

### Customer message:
{text}

### Label:
"""


class IntentClassifier:
    def __init__(self, model_path: str):
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        self.checkpoint = config.get("model_checkpoint", "outputs/banking-intent-model")
        self.max_seq_length = config.get("max_seq_length", 512)

        labels_path = os.path.join(self.checkpoint, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.json not found in {self.checkpoint}")
        with open(labels_path) as f:
            self.valid_labels = json.load(f)
        print(f"[INFO] Loaded {len(self.valid_labels)} valid labels")

        # Pre-build label string cho prompt
        self.labels_str = "\n".join(f"- {lb}" for lb in self.valid_labels)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=config.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)

        # Pre-tokenize tất cả labels để tăng tốc inference
        # Mỗi label được tokenize KHÔNG có prefix space (add_special_tokens=False)
        self._label_ids = []
        for lb in self.valid_labels:
            ids = self.tokenizer.encode(lb, add_special_tokens=False)
            self._label_ids.append(ids)
        print(f"[INFO] Pre-tokenized {len(self._label_ids)} labels")

    def _compute_label_logprob(self, prompt_ids: torch.Tensor, label_ids: list[int]) -> float:
        """
        Tính tổng log-prob của label_ids khi append vào prompt_ids.
        Dùng teacher forcing: input = [prompt + label], target shifted by 1.
        """
        label_tensor = torch.tensor(label_ids, dtype=torch.long, device="cuda")
        full_ids = torch.cat([prompt_ids[0], label_tensor]).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids=full_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Chỉ lấy logits ở vị trí tương ứng với label tokens
        prompt_len = prompt_ids.shape[1]
        label_logits = logits[0, prompt_len - 1 : prompt_len - 1 + len(label_ids), :]
        log_probs = torch.nn.functional.log_softmax(label_logits, dim=-1)

        # Sum log-prob của từng token trong label
        score = sum(
            log_probs[i, label_ids[i]].item()
            for i in range(len(label_ids))
        )
        # Normalize theo độ dài label để tránh bias ngắn/dài
        return score / len(label_ids)

    def __call__(self, message: str) -> str:
        prompt = PROMPT_TEMPLATE.format(labels=self.labels_str, text=message)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        # Score tất cả labels và pick argmax
        best_label = None
        best_score = float("-inf")

        for lb, lb_ids in zip(self.valid_labels, self._label_ids):
            score = self._compute_label_logprob(prompt_ids, lb_ids)
            if score > best_score:
                best_score = score
                best_label = lb

        del prompt_ids
        torch.cuda.empty_cache()
        return best_label


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

        wrong = [(y_true_clean[i], y_pred_clean[i]) for i in range(total) if y_true_clean[i] != y_pred_clean[i]]
        if wrong:
            from collections import Counter
            print("\n--- TOP 10 MOST CONFUSED PAIRS ---")
            for (true, pred), count in Counter(wrong).most_common(10):
                print(f"  TRUE: '{true}' → PRED: '{pred}' ({count}x)")

    except FileNotFoundError:
        print("Error: sample_data/test.csv not found.")