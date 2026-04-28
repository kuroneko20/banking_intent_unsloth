import json
import os
import yaml
import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# GIỐNG HỆT train.py
PROMPT_TEMPLATE = """### Instruction:
Classify the banking intent. Reply with ONLY the intent label, nothing else.

### Input:
{}

### Response:
"""


def get_embedding(model, tokenizer, text: str) -> np.ndarray:
    """Lấy mean-pooling embedding từ hidden state cuối của model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Mean pool last hidden state
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return emb[0].cpu().float().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class IntentClassification:
    def __init__(self, model_path):
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

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=config.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)

        # Pre-compute embeddings cho tất cả labels một lần
        print("[INFO] Pre-computing label embeddings...")
        self.label_embeddings = {}
        for label in self.valid_labels:
            # Embed label dưới dạng readable text: "activate my card"
            readable = label.replace("_", " ")
            self.label_embeddings[label] = get_embedding(self.model, self.tokenizer, readable)
        print(f"[INFO] Done — {len(self.label_embeddings)} label embeddings ready")

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
        3-bước matching:
        1. Exact match → trả về ngay
        2. Substring match → trả về ngay  
        3. Semantic similarity dùng model embeddings
        """
        first_line = raw.strip().splitlines()[0].strip()
        cleaned = first_line.strip(".,;:!?\"'() ").lower()

        # 1. Exact match
        for label in self.valid_labels:
            if label.lower() == cleaned:
                return label

        # 2. Substring: label nằm trong output
        for label in self.valid_labels:
            if label.lower() in cleaned:
                return label

        # 3. Semantic similarity — embed raw output, tìm label gần nhất
        raw_emb = get_embedding(self.model, self.tokenizer, first_line.replace("_", " "))
        best_label = None
        best_score = -1.0
        for label, label_emb in self.label_embeddings.items():
            score = cosine_sim(raw_emb, label_emb)
            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    def __call__(self, message: str) -> str:
        prompt = PROMPT_TEMPLATE.format(message)
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
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predicted = self._find_best_label(raw)

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
        print("Error: sample_data/test.csv not found.")