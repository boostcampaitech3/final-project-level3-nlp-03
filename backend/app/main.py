from typing import List
from unicodedata import name
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form


from util.read_input import preprocess

app = FastAPI()

class StudentAnswer(BaseModel):
    student_id : str
    answer : str

class Problem(BaseModel):
    question: str
    gold_answer: str
    keywords: list
    answers : List[StudentAnswer]


class ProblemList(BaseModel):
    data: List[Problem]


@app.get("/api/")
def read_root():
    return "hello gompada"


@app.post("/api/input")
def read_item(data : ProblemList):

    a = preprocess(data)

    return data




#initial
# uvicorn main:app --host=0.0.0.0 --port=8000 --reload