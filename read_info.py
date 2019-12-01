
import json
from pydantic import BaseModel
from typing import List


class Answer(BaseModel):
    answer: str
    extra_data: dict

class Question(BaseModel):
    text: str
    answers: List[Answer]
    id:str

def load_file(path:str)->List[Question]:
    return [Question(**q) for q in json.load(open(path))]


if __name__ == "__main__":
    load_file('saved.json')