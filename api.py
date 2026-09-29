from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

app = FastAPI()

language_model_path = "checkpoints/model.pth"
language_model_creation_time = int(os.path.getmtime(language_model_path))

# V1 Models Response
# Target: {"object": "list", "data": [{"id": "bobgpt", "object": "model", "owned_by": "you"}]}

class V1Models(BaseModel):
    object: str
    data: list 

@app.get("/v1/models", response_model=V1Models)
def v1_models() -> V1Models:
    data = [{"id": "bobgpt", "object": "model", "owned_by": "Robert Cowher", "created": language_model_creation_time}]

    return V1Models(object="list", data=data, queue_depth=3)

# V1 Chat Completions

class V1ChatCompletionsRequest(BaseModel):
    model: str
    messages: list
    stream: bool
    temperature: float
    max_tokens: int

class V1ChatCompletionsResponse(BaseModel):
    pass

@app.post("/v1/chat/completions", response_model=V1ChatCompletionsResponse, status_code=201)
def get_chat_completion(req: V1ChatCompletionsRequest):
    
    



