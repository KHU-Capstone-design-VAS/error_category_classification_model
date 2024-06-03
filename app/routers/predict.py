from fastapi import APIRouter, HTTPException
from schemas.request import TextData
from services.classification_prediction import classification_predict
from services.question_answering_prediction import question_answering_predict
from utils.make_question import make_question

router = APIRouter()

@router.post("/predict/")
async def get_prediction(data: TextData):
    text_input = f"[original]{data.origin}[/original][summary]{data.summary}"
    try:
        predicted_label = classification_predict(text_input)
        question = make_question(predicted_label)
        if question:
            question_answering_predict_result = question_answering_predict(text_input, question)
            answer = question_answering_predict_result['answer']
            start_index = question_answering_predict_result['start_index']
            end_index = question_answering_predict_result['end_index']
            return {
                    "predicted_label": predicted_label,
                    "question" : question,
                    "answer" : answer,
                    "start_index" : start_index,
                    "end_index" : end_index,
                    }
        else:
            return {
                "predicted_label": predicted_label,
                "question" : question,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
