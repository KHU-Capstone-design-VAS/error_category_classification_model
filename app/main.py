from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import predict

app = FastAPI()

origins = [
    "http://localhost/*",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], )

app.include_router(predict.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

    from fastapi import FastAPI
