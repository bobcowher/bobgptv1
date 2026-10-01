from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
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
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int = 100

class V1ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int 
    model: str = "bobgpt"
    choices: list
    usage: dict 

@app.post("/v1/chat/completions", response_model=V1ChatCompletionsResponse, status_code=200)
def get_chat_completion(req: V1ChatCompletionsRequest):
    choices = [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from bobgpt!",
                        "refusal": None
                        },
                    "logprobs": None,
                    "finish_reason": "stop"
                } 
            ]
    usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
            } 

    response = V1ChatCompletionsResponse(id="5",
                                         object="chat.completion",
                                         created=int(time.time()),
                                         model="bobgpt",
                                         choices=choices,
                                         usage=usage
                                         )
    return response



# {
#   "id": "chatcmpl-abc123",
#   "object": "chat.completion",
#   "created": 1790640000,
#   "model": "bobgpt",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Hello from bobgpt!",
#         "refusal": null
#       },
#       "logprobs": null,
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 10,
#     "completion_tokens": 5,
#     "total_tokens": 15
#   }
# }


    
    



