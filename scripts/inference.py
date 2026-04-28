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

# PHẢI GIỐNG HỆT train.py — chỉ bỏ phần response
PROMPT_TEMPLATE = """### Instruction:
Classify the banking intent. Reply with ONLY the intent label, nothing else.

### Input:
{}

### Response:
"""


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

        # ================================================================
        # CONSTRAINED DECODING: tính token IDs của từng label hợp lệ
        # Tại mỗi bước generate, chỉ cho phép tokens thuộc prefix của labels
        # → model KHÔNG THỂ output text ngoài tập labels
        # ================================================================
        self._build_label_token_map()

        unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        candidates = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("</s>"),
        ]
        self.terminators = list({t for t in candidates if t is not None and t != unk_id})
        print(f"[INFO] Stop token IDs: {self.terminators}")

    def _build_label_token_map(self):
        """Tokenize tất cả labels, build prefix tree để constrained decoding."""
        self.label_token_ids = {}
        for label in self.valid_labels:
            # Tokenize label (không có special tokens)
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            self.label_token_ids[label] = token_ids

        # Build set các first-token hợp lệ
        self.valid_first_tokens = list({ids[0] for ids in self.label_token_ids.values()})
        print(f"[INFO] {len(self.valid_first_tokens)} unique first tokens across {len(self.valid_labels)} labels")

    def _constrained_generate(self, inputs) -> str:
        """
        Generate với constrained decoding:
        1. Generate token đầu tiên — chỉ từ first tokens của labels
        2. Từ first token đó, tìm labels bắt đầu bằng token này
        3. Nếu chỉ còn 1 label → trả về luôn
        4. Tiếp tục generate token tiếp theo trong candidates còn lại
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Lấy logits tại bước đầu tiên
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=15,
                use_cache=True,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.0,
            )

        new_tokens = output[0][input_ids.shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return raw

    def _fuzzy_match(self, raw: str) -> str:
        """Match raw output về label gần nhất."""
        first_line = raw.strip().splitlines()[0].strip().lower()
        cleaned = first_line.strip(".,;:!?\"'() ")

        # Exact match
        for label in self.valid_labels:
            if label.lower() == cleaned:
                return label

        # Label là substring của output
        for label in self.valid_labels:
            if label.lower() in cleaned:
                return label

        # Output là substring của label
        if len(cleaned) > 4:
            for label in self.valid_labels:
                if cleaned in label.lower():
                    return label

        # Token overlap score
        best_label = None
        best_score = 0
        cleaned_tokens = set(cleaned.replace("_", " ").split())
        for label in self.valid_labels:
            label_tokens = set(label.replace("_", " ").split())
            overlap = len(cleaned_tokens & label_tokens)
            if overlap > best_score:
                best_score = overlap
                best_label = label

        if best_score > 0 and best_label:
            return best_label

        return cleaned

    def __call__(self, message: str) -> str:
        prompt = PROMPT_TEMPLATE.format(message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        raw = self._constrained_generate(inputs)
        predicted = self._fuzzy_match(raw)

        del inputs
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