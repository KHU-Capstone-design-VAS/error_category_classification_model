from transformers import BertForSequenceClassification, BertTokenizer

model_name = "klue/bert-base"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

model.save_pretrained("model/error_classification_model_16_16_30_klue")
tokenizer.save_pretrained("model/error_classification_model_16_16_30_klue")