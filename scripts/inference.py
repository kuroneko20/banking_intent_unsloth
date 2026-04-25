import yaml
import torch
import warnings
from unsloth import FastLanguageModel

# Tắt toàn bộ các cảnh báo (Warnings) cho giao diện console sạch đẹp
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
        # Ép mô hình chỉ trả lời bằng tên nhãn (exact label)
        prompt = f"Classify the banking intent of the following text. Output ONLY the intent label name, nothing else.\nText: {message}\nIntent:"
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        # Xóa bỏ tham số max_length ẩn gây xung đột warning, chỉ giữ max_new_tokens
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=15, 
            use_cache=True, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predicted_label = response.split("Intent:")[-1].strip()
        return predicted_label

if __name__ == "__main__":
    print("Initializing Model...")
    classifier = IntentClassification(model_path="configs/inference.yaml")
    
    test_messages = [
        "I lost my card yesterday, please help me block it.",
        "What is the exchange rate for USD to EUR?",
        "Why was I charged an extra fee for my ATM withdrawal?"
    ]
    
    print("\n--- INFERENCE RESULTS ---")
    for msg in test_messages:
        intent = classifier(message=msg)
        print(f"Input: {msg}")
        print(f"Predicted Intent: {intent}\n")