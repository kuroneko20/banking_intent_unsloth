"""
SetFit inference cho Banking77.
Load model đã train, predict intent từ text input.
"""
import json
import os
import yaml
import pandas as pd
from setfit import SetFitModel
from sklearn.metrics import accuracy_score


class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        self.checkpoint = config.get("model_checkpoint", "outputs/banking-intent-model")

        # Load label mapping
        labels_path = os.path.join(self.checkpoint, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.json not found in {self.checkpoint}")
        with open(labels_path) as f:
            raw = json.load(f)
        # Support cả 2 format: {0: "label"} hoặc ["label", ...]
        if isinstance(raw, dict):
            self.id2label = {int(k): v for k, v in raw.items()}
        else:
            self.id2label = {i: v for i, v in enumerate(raw)}
        print(f"[INFO] Loaded {len(self.id2label)} labels")

        print("[INFO] Loading SetFit model...")
        self.model = SetFitModel.from_pretrained(self.checkpoint)
        print("[INFO] Model ready")

    def __call__(self, message: str) -> str:
        pred_id = self.model.predict([message])[0]
        # SetFit có thể trả về label name trực tiếp hoặc integer
        if isinstance(pred_id, str):
            return pred_id
        return self.id2label.get(int(pred_id), str(pred_id))

    def predict_batch(self, messages: list) -> list:
        preds = self.model.predict(messages)
        results = []
        for p in preds:
            if isinstance(p, str):
                results.append(p)
            else:
                results.append(self.id2label.get(int(p), str(p)))
        return results


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

        print(f"Predicting {len(texts)} samples (batch mode)...")
        # SetFit predict batch cùng lúc — nhanh hơn nhiều so với LLM
        y_pred = classifier.predict_batch(texts)

        y_true_clean = [str(y).strip().lower() for y in y_true]
        y_pred_clean = [str(y).strip().lower() for y in y_pred]
        acc = accuracy_score(y_true_clean, y_pred_clean)

        print("\n==========================================")
        print(f"✅ FINAL ACCURACY ON TEST SET: {acc * 100:.2f}%")
        print("==========================================")

        print("\n--- DEBUG: First 10 predictions ---")
        for i in range(min(10, len(y_true))):
            mark = "✅" if y_true_clean[i] == y_pred_clean[i] else "❌"
            print(f"{mark} TRUE: '{y_true_clean[i]}' | PRED: '{y_pred_clean[i]}'")

    except FileNotFoundError:
        print("Error: sample_data/test.csv not found. Run preprocess_data.py first.")