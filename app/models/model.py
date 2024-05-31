import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "../model/error_classification_model_16_16_30_klue"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()