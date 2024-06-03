from transformers import BertTokenizer, BertForQuestionAnswering

question_answering_model_path = "./model/question-answering-temp"
question_answering_tokenizer = BertTokenizer.from_pretrained(question_answering_model_path)
question_answering_model = BertForQuestionAnswering.from_pretrained(question_answering_model_path)

question_answering_model.eval()