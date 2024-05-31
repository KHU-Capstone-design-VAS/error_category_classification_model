from fastapi import APIRouter, HTTPException
from schemas.request import TextData
from services.prediction import predict

router = APIRouter()

@router.post("/predict/")
async def get_prediction(data: TextData):
    text_input = f"[original]{data.origin}[/original][summary]{data.summary}"
    try:
        predicted_label = predict(text_input)
        return {"predicted_label": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
