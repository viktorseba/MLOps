from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
from pydantic import BaseModel
import re
from fastapi import UploadFile, File
from typing import Optional
import cv2
from fastapi.responses import FileResponse

app = FastAPI()


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


class Mail(BaseModel):
    email: str
    domain: str


# @app.get("/restric_items/{item_id}")
# def read_item(item_id: ItemEnum):
#     return {"item_id": item_id}


database = {"username": [], "password": []}


@app.post("/login/")
def login(username: str, password: str):
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open("database.csv", "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/text_model/")
def contains_email(data: Mail):
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None,
    }
    return response


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
    img = cv2.imread(data.filename)
    res = cv2.resize(img, (h, w))
    cv2.imwrite("image_resize.jpg", res)
    # FileResponse('image_resize.jpg')

    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    # return response
    return FileResponse("image_resize.jpg")


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


if __name__ == "__main__":
    app.run()


# uvicorn --reload --port 8000 main:app
