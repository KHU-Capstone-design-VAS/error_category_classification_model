from typing import Dict, Any

import torch
from question_answering_model import question_answering_tokenizer, question_answering_model


# text_input : origin text, question : make_question 함수를 통해 나온 question
def question_answering_predict(text_input: str, question: str) -> dict[
    str, int | int | float | float | bool | bool | Any]:
    inputs = question_answering_tokenizer.encode_plus(
        question,
        text_input,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = question_answering_model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits) + 1
    answer = question_answering_tokenizer.convert_tokens_to_string(
        question_answering_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))

    return {
        "answer": answer,
        "start_index": start_index.item(),
        "end_index": end_index.item()
    }
