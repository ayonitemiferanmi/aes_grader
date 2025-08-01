from typing import Union
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from huggingface_hub import InferenceClient, hf_hub_download
import os
from dotenv import load_dotenv
import numpy as np
# import pickle as pkl
import requests
import joblib

load_dotenv()

# Creating an instance of the fastapi
app = FastAPI()

# using requests to access the model
API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-base/pipeline/feature-extraction"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Loading the model from huggingface hub
REPO_ID = "Ayonitemi-Feranmi/aes_grader"
FILENAME = "aes_grader.pkl"

USE_OFFLINE = True

# if USE_OFFLINE:
#     model_path = "/model/aes_grader.pkl"
#     model = joblib.load(model_path)
# else:
model = joblib.load(
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=os.getenv("HF_TOKEN"))
    )

# with open(r"C:\Users\Rise Networks\Desktop\AI_theory_grader\aes_grader.pkl", "rb") as f:
#     aes_grader = pkl.load(f)

# creating a BaseModel class to take in the question data info
class Exam_Data(BaseModel):
    question_id: str
    type: str
    answer: str
    correct: str
    status: str  


@app.get("/")
def read_root():
    return "Hello World!"


@app.post("/score_result")
def return_score(data: Exam_Data):
    # receiving the input dictionary
    json_data = data.dict()

    # cleaning up the user answer
    json_data["answer"] = json_data["answer"].strip("</p>")

    # asserting that the question passed is a theory question
    if json_data["type"].lower() == "theory":
        source_embedding = np.mean(query({"inputs": f"{json_data['correct']}"}), axis=0)
        compare_embedding = np.mean(query({"inputs": f"{json_data['answer']}"}), axis=0)

        # input vector:finding the difference between the two embedding while reshaping to the attention structure
        input_vector = np.array(np.abs(source_embedding, compare_embedding)).reshape(1, 1, 768)

        # making predictions with the aes_grader model
        score = model.predict(input_vector)
        print(score)
        # creating a threshold for the similarity score
        if score >= 0.5:
            json_data["passed"] = True
        else:
            json_data["passed"] = False
    
    return json_data



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)