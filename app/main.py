from transformers import pipeline
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


class Email(BaseModel):
    body: str


@app.on_event("startup")
def load_model():

    global sentiment_pipeline_default, sentiment_pipeline_emotions

    trained_model = "bhadresh-savani/distilbert-base-uncased-emotion"

    sentiment_pipeline_default = pipeline("sentiment-analysis")

    sentiment_pipeline_emotions = pipeline("sentiment-analysis", model=trained_model)


@app.get("/api/v1/sentiment")
def sentiment(email: Email):

    res = sentiment_pipeline_default(email.body)

    return res


@app.get("/api/v1/emotions")
def emotions(email: Email):

    res = sentiment_pipeline_emotions(email.body)

    return res


@app.get('/')
def hello_there():
    return 'Hello from MLOps Team 5!'


