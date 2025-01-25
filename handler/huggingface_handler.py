# huggingface_handler.py
import time
import xxhash
from fastapi.responses import StreamingResponse
import librosa
import numpy as np
import base64 
import torch
from entity.entity import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    UsageInfo,
)
import json
from handler.base_handler import BaseHandler
from transformers import AutoModel

from utils import get_either, get_file_from_any
class HuggingfaceHandler(BaseHandler):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.prev_outs_cache = {}

    def _hash_message_content(self, messages):
        hasher = xxhash.xxh32()
        
        for message in messages:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "audio":
                        audio_str = content["audio_url"].split("base64,")[1]
                        audio_bytes = base64.b64decode(audio_str)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                        hasher.update(bytes(audio_array))
                    elif content["type"] == "text":
                        hasher.update(content["text"].encode())
            else:
                hasher.update(str(message["content"]).encode())
                
        return hasher.hexdigest()

    @torch.no_grad()
    def generate_stream(self, request):
        messages = request.messages
        print(f"Len of messages: {len(messages)}")
        
        curr_hash = self._hash_message_content(messages[-2:])
        prev_hash = self._hash_message_content(messages[-4:-2]) if len(messages) > 2 else None

        prev_outs = self.prev_outs_cache.get(prev_hash) if prev_hash else None
        
        # process current audio
        for content in messages[-1]["content"]:
            if content["type"] == "audio":
                audio_str = content["audio_url"].split("base64,")[1]
                audio_bytes = base64.b64decode(audio_str)
                current_audio = np.frombuffer(audio_bytes, dtype=np.float32)
                current_audio = current_audio / np.max(np.abs(current_audio))
                break
        
        # response
        results_generator = self.model.generate_stream(
            current_audio,
            text_prompt=(
                "You are a voice assistant. Your interface with users will be voice. "
                "You should use short and concise responses. You should tailor your response "
                "style to speech using spelled out numbers and avoiding all formatting except "
                "for standard punctuation."
            ) if prev_outs is None else None,
            do_sample=request.temperature > 0.005,
            max_new_tokens=request.max_tokens,
            init_outputs=prev_outs,
            return_outputs=True,
        )

        @torch.no_grad()
        def stream_results():
            prev_output = ""
            for text_output, outs in results_generator:
                delta_text = text_output[len(prev_output):]
                response = {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta_text},
                        "finish_reason": None
                    }]
                }
                yield "data: " + json.dumps(response) + "\n\n"
                prev_output = text_output

            self.prev_outs_cache[curr_hash] = outs

            yield "data: " + json.dumps({
                "id": f"chatcmpl-{time.time()}",
                "object": "chat.completion.chunk", 
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }) + "\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")

    def generate(self, request):
        raise NotImplementedError