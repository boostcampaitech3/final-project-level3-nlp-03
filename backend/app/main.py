from typing import List
from unicodedata import name
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

class Problem(BaseModel):
    question: str
    answer: str
    keywords: list



class ProblemList(BaseModel):
    data: List[Problem]


@app.get("/")
def read_root():
    return "hello gompada"


@app.post("/input")
def read_item(data : ProblemList):

  return data

@app.post("/uploadfiles/")
async def create_upload_files(
    files: List[UploadFile] = File(description="Multiple files as UploadFile"),
):
    return {"filenames": [file.filename for file in files]}


#initial
# uvicorn main:app --host=0.0.0.0 --port=8000 --reload