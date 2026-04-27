import yaml
import torch
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel

# Tắt các cảnh báo thừa để màn hình console sạch đẹp
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.checkpoint = config.get("model_checkpoint", "outputs/banking-intent-model")
        self.max_seq_length = config.get("max_seq_length", 256)
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=config.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)
        
    def __call__(self, message):
        prompt = f"Classify the banking intent of the following text.\nText: {message}\nIntent:"
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Bọc trong no_grad để không lưu thông tin huấn luyện, tiết kiệm RAM
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=15, 
                use_cache=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predicted_label = response.split("Intent:")[-1].strip()
        
        # XÓA TENSOR VÀ DỌN DẸP BỘ NHỚ GPU SAU MỖI CÂU
        del inputs, outputs
        torch.cuda.empty_cache()
        
        return predicted_label

if __name__ == "__main__":
    print("Initializing Model...")
    classifier = IntentClassification(model_path="configs/inference.yaml")
    
    # --- PHẦN 1: DEMO VÀI CÂU MẪU ---
    test_messages = [
        "I lost my card yesterday, please help me block it.",
        "What is the exchange rate for USD to EUR?",
        "Why was I charged an extra fee for my ATM withdrawal?"
    ]
    
    print("\n--- 1. INFERENCE DEMO ---")
    for msg in test_messages:
        intent = classifier(message=msg)
        print(f"Input: {msg}")
        print(f"Predicted Intent: {intent}\n")


    # --- PHẦN 2: ĐÁNH GIÁ ACCURACY TRÊN TEST SET ---
    print("\n--- 2. EVALUATING ON TEST SET ---")
    print("Loading sample_data/test.csv...")
    try:
        df_test = pd.read_csv("sample_data/test.csv")
        y_true = df_test["intent"].tolist()
        texts = df_test["text"].tolist()
        y_pred = []
        
        total_samples = len(texts)
        print(f"Predicting {total_samples} samples... (This may take a few minutes)")
        
        for i, text in enumerate(texts):
            pred = classifier(message=text)
            y_pred.append(pred)
            
            # In tiến trình sau mỗi 20 câu để dễ theo dõi
            if (i + 1) % 20 == 0 or (i + 1) == total_samples:
                print(f"  -> Processed {i + 1}/{total_samples} samples...")
                
        # Tính toán độ chính xác (Accuracy)
        acc = accuracy_score(y_true, y_pred)
        
        print("\n==========================================")
        print(f"✅ FINAL ACCURACY ON TEST SET: {acc * 100:.2f}%")
        print("==========================================")
        
    except FileNotFoundError:
        print("Error: Could not find sample_data/test.csv. Please run preprocess_data.py first.")