from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import os
import sys
import uuid
from languagemodel import *
from chat_template import *
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.responses import StreamingResponse
from config import GPT_CONFIG_124M
from text_stream import TextStream


app = FastAPI()

language_model_path = "checkpoints/posttrain/model.pth"
language_model_creation_time = int(os.path.getmtime(language_model_path))


model = LanguageModel(gpt_config=GPT_CONFIG_124M)
model.load_the_model(language_model_path)
model.model.eval()

context_size = model.model.pos_emb.weight.shape[0]
eot = model.tokenizer.eot_token


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
    max_completion_tokens: int | None = None   # newer name for max_tokens; wins when both are sent

class V1ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int 
    model: str = "bobgpt"
    choices: list
    usage: dict = None 


@app.post("/v1/chat/completions", response_model=V1ChatCompletionsResponse, status_code=200)
def get_chat_completion(req: V1ChatCompletionsRequest):
    if req.max_completion_tokens is not None:
        req.max_tokens = req.max_completion_tokens

    if req.stream:
        return StreamingResponse(get_chat_completion_streaming(req), media_type="text/event-stream")
    else:
        return get_chat_completion_block(req)


def get_chat_completion_streaming(req: V1ChatCompletionsRequest):
    total_tokens = 0
    stop_reason = None
    id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    encoded = get_encoded_messages(req.messages)

    chunk = get_chunk(id=id,
                      created=created,
                      model=req.model,
                      delta={"role": "assistant", "content": ""},
                      stop_reason=stop_reason)

    yield f"data: {chunk}\n\n"

    text_stream = TextStream(model.tokenizer)

    # finally also runs when the client disconnects mid-stream (Stop in Open WebUI):
    # the generator is closed at its paused yield and the code after the loop never runs.
    try:
        with torch.no_grad():
            for token_id in model.generate_streaming(
                                idx = encoded,
                                max_new_tokens=req.max_tokens,
                                context_size=context_size,
                                temperature=req.temperature,
                                top_k=40,
                                eos_id=eot):
                total_tokens += 1

                # Text that's safe to send: stop markers and trailing whitespace are held back.
                completion = text_stream.push(token_id.item())

                if completion:
                    chunk = get_chunk(id=id,
                                      created=created,
                                      model=req.model,
                                      delta={"content": completion},
                                      stop_reason=None)

                    yield f"data: {chunk}\n\n"

                if text_stream.stopped:
                    stop_reason = "stop"
                    break

            # Generation ended without a stop marker: send what was held back.
            completion = text_stream.finish()
            if completion:
                chunk = get_chunk(id=id,
                                  created=created,
                                  model=req.model,
                                  delta={"content": completion},
                                  stop_reason=None)

                yield f"data: {chunk}\n\n"

            # No stop text: either the token budget ran out or the model emitted EOT.
            if stop_reason is None:
                stop_reason = "length" if total_tokens == req.max_tokens else "stop"

            chunk = get_chunk(id=id,
                              created=created,
                              model=req.model,
                              delta={},
                              stop_reason=stop_reason)

            yield f"data: {chunk}\n\n"
        
            yield "data: [DONE]\n\n"
    finally:
        # Hand cached GPU memory back so idle bobgpt doesn't crowd gemma on the 3060.
        torch.cuda.empty_cache()
            


def get_chat_completion_block(req: V1ChatCompletionsRequest):

    stop_reason = "length"
    encoded = get_encoded_messages(req.messages)
    created = int(time.time()) 
    id = f"chatcmpl-{uuid.uuid4().hex}"

    with torch.no_grad():
        token_ids = model.generate(
                idx = encoded,
                max_new_tokens=req.max_tokens,
                context_size=context_size,
                temperature=req.temperature,
                top_k=40,
                eos_id=eot
                )
    torch.cuda.empty_cache()  # see get_chat_completion_streaming

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

    response = V1ChatCompletionsResponse(id=id,
                                         object="chat.completion",
                                         created=created,
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


def get_chunk(id, created, model, delta, stop_reason):

        choices = [
                    {
                        "index": 0,
                        "delta": delta, 
                        "logprobs": None,
                        "finish_reason": stop_reason
                    } 
                ]

        chunk = V1ChatCompletionsResponse(id=id,
                                        object="chat.completion.chunk",
                                        created=created,
                                        model=model,
                                        choices=choices
                                        )

        return chunk.model_dump_json()

