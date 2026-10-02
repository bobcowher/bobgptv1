from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import os
import sys
from languagemodel import *
from chat_template import *
from config import GPT_CONFIG_124M

# messages = [{"role": "system", "content": "You are bobgpt."},
#             {"role": "user", "content": "What is a tuple?"},
#             {"role": "assistant", "content": "An immutable sequence."},
#             {"role": "user", "content": "Why use one?"}]
#
# chat = render_prompt(messages)  


app = FastAPI()

language_model_path = "checkpoints/model.pth"
language_model_creation_time = int(os.path.getmtime(language_model_path))


model = LanguageModel(gpt_config=GPT_CONFIG_124M)
model.load_the_model("checkpoints/model.pth")
model.model.eval()

context_size = model.model.pos_emb.weight.shape[0]
eot = model.tokenizer.eot_token


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

    messages = render_prompt(req.messages) 

    encoded = model.text_to_token_ids(messages, model.tokenizer).to(model.device)
    with torch.no_grad():
        token_ids = model.generate(
                idx = encoded,
                max_new_tokens=req.max_tokens,
                context_size=context_size,
                temperature=req.temperature,
                top_k=40,
                eos_id=eot
                )
    
    completion = model.token_ids_to_text(token_ids[:, encoded.shape[1]:], model.tokenizer)
    
    usage = {
            "prompt_tokens": len(encoded),
            "completion_tokens": len(completion),
            "total_tokens": len(encoded) + len(completion)
            } 
    
    choices = [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion,
                        "refusal": None
                        },
                    "logprobs": None,
                    "finish_reason": "stop"
                } 
            ]

    response = V1ChatCompletionsResponse(id="5",
                                         object="chat.completion",
                                         created=int(time.time()),
                                         model=req.model,
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


    
    



