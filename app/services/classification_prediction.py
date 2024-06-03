import torch
from models.classification_model import classification_tokenizer, classification_model

def classification_predict(text: str) -> int:
    inputs = classification_tokenizer.encode_plus(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label
