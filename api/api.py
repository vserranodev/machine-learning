from fastapi import FastAPI, Response
from pydantic import BaseModel
from ml_models.scratch.models import SequentialModel

app = FastAPI()

@app.get("/")
def root():
    return "API ON DEVELOPMENT"

@app.get("/")
def root():
    pass

@app.post("/execute")
def execute():
    pass





