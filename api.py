from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import os
import sys
from languagemodel import *
from chat_template import *
from fastapi.sse import EventSourceResponse, ServerSentEvent
from config import GPT_CONFIG_124M


app = FastAPI()

language_model_path = "checkpoints/model.pth"
language_model_creation_time = int(os.path.getmtime(language_model_path))


model = LanguageModel(gpt_config=GPT_CONFIG_124M)
model.load_the_model("checkpoints/model.pth")
model.model.eval()

context_size = model.model.pos_emb.weight.shape[0]
eot = model.tokenizer.eot_token

### Just here for the POC

# encoded = model.text_to_token_ids(messages, model.tokenizer).to(model.device)
#
# with torch.no_grad():
#     for token_id in model.generate_streaming(
#                         idx = encoded,
#                         max_new_tokens=50,
#                         context_size=context_size,
#                         temperature=0,
#                         top_k=40,
#                         eos_id=eot):
#
#         completion = model.token_ids_to_text(token_id, model.tokenizer)
#         print(completion)


### Just here for the POC

# V1 Models Response
# Target: {"object": "list", "data": [{"id": "bobgpt", "object": "model", "owned_by": "you"}]}

# sys.exit(1)

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
    usage: dict = None 


@app.post("/v1/chat/completions", response_model=V1ChatCompletionsResponse, status_code=200)
def get_chat_completion(req: V1ChatCompletionsRequest):
    if req.stream:
        return get_chat_completion_streaming(req) 
    else:
        return get_chat_completion_block(req)


def get_chat_completion_streaming(req: V1ChatCompletionsRequest):
    stop = False
    total_tokens = 0
    stop_reason = None

    encoded = get_encoded_messages(req.messages)

    with torch.no_grad():
        for token_id in model.generate_streaming(
                            idx = encoded,
                            max_new_tokens=req.max_tokens,
                            context_size=context_size,
                            temperature=req.temperature,
                            top_k=40,
                            eos_id=eot):
            total_tokens += 1

            completion = model.token_ids_to_text(token_id, model.tokenizer)
    
            # See if we have a text stop. 
            end_text_idx = completion.find("### End")
            question_text_idx = completion.find("\n### Question")

            if((end_text_idx != -1) or (question_text_idx != -1)):
                stop = True 
                stop_reason = "stop" 

            if total_tokens > req.max_tokens:
                stop = True
                stop_reason = "length"


            choices = [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": completion,
                                "refusal": None
                                },
                            "logprobs": None,
                            "finish_reason": stop_reason
                        } 
                    ]

            if stop:
                yield ServerSentEvent(raw_data="[DONE]")

            yield V1ChatCompletionsResponse(id="5",
                                            object="chat.completion",
                                            created=int(time.time()),
                                            model=req.model,
                                            choices=choices
                                            )
            


def get_chat_completion_block(req: V1ChatCompletionsRequest):

    stop_reason = "length"

    encoded = get_encoded_messages(req.messages)

    with torch.no_grad():
        token_ids = model.generate(
                idx = encoded,
                max_new_tokens=req.max_tokens,
                context_size=context_size,
                temperature=req.temperature,
                top_k=40,
                eos_id=eot
                )


    completion_tokens = token_ids[:, encoded.shape[1]:]

    completion = model.token_ids_to_text(completion_tokens, model.tokenizer)

    # Strip out End
    end_text_idx = completion.find("### End")

    if(end_text_idx != -1):
        completion = completion[:end_text_idx]
        stop_reason = "stop"
    
    # If no end is found, strip out question.
    question_text_idx = completion.find("\n### Question")

    if(question_text_idx != -1):
        completion = completion[:question_text_idx]
        stop_reason = "stop"

    # Remove whitespace
    completion = completion.strip()

    if completion_tokens.shape[1] < req.max_tokens:
        stop_reason = "stop"


    usage = {
            "prompt_tokens": encoded.shape[1],
            "completion_tokens": completion_tokens.shape[1],
            "total_tokens": encoded.shape[1] + completion_tokens.shape[1]
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
                    "finish_reason": stop_reason
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


def get_encoded_messages(req_messages):
    # content is a string, or a list of parts like {"type": "text", "text": "..."}; keep only the text.
    normalized = []
    for message in req_messages:
        content = message["content"]
        if not isinstance(content, str):
            content = "".join(part["text"] for part in content if part.get("type") == "text")
        normalized.append({"role": message["role"], "content": content})

    messages = render_prompt(normalized)

    encoded = model.text_to_token_ids(messages, model.tokenizer).to(model.device)

    return encoded
