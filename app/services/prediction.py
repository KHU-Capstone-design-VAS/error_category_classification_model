from models.model import tokenizer, model
import torch

def predict(text: str) -> int:
    inputs = tokenizer.encode_plus(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label
