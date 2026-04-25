import yaml
import torch
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, model_path):
        """
        Khởi tạo model từ config file[cite: 60, 64].
        """
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.checkpoint = config.get("model_checkpoint", "outputs/banking-intent-model")
        self.max_seq_length = config.get("max_seq_length", 256)
        
        # Load model & tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=config.get("load_in_4bit", True),
        )
        # Bật chế độ tối ưu hóa suy luận (Inference mode gấp 2x tốc độ)
        FastLanguageModel.for_inference(self.model)
        
    def __call__(self, message):
        """
        Dự đoán intent từ input message[cite: 61, 69].
        """
        prompt = f"Classify the banking intent of the following text.\nText: {message}\nIntent:"
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=20, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # Tách phần kết quả Intent được model sinh ra
        predicted_label = response.split("Intent:")[-1].strip()
        return predicted_label

# === Ví dụ sử dụng (Demo) [cite: 70] ===
if __name__ == "__main__":
    print("Initializing Model...")
    classifier = IntentClassification(model_path="configs/inference.yaml")
    
    # Test cases
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