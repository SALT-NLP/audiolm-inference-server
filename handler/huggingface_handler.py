# huggingface_handler.py
import time
import xxhash
from fastapi.responses import StreamingResponse
import librosa
import numpy as np
import base64  # Add this import
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

    @torch.no_grad()
    def generate_stream(self, request):
        audio_data = []
        messages = request.messages
        
        # Extract audio data from messages
        for message in messages:
            if message["role"] == "user" and isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "audio":
                        # Extract base64 audio data
                        audio_str = content["audio_url"].split("base64,")[1]
                        audio_bytes = base64.b64decode(audio_str)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                        audio_data.append(audio_array)

        # Get current audio
        current_audio = audio_data[-1]
        curr_hash = xxhash.xxh32(bytes(current_audio)).hexdigest()

        # Get previous outputs if they exist
        prev_outs = None
        if len(audio_data) > 1:
            prev_audio = audio_data[-2]
            prev_hash = xxhash.xxh32(bytes(prev_audio)).hexdigest()
            prev_outs = self.prev_outs_cache.get(prev_hash)

        # Normalize audio
        current_audio = current_audio / np.max(np.abs(current_audio))

        # Generate response
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
                # print("generating response")
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

            # Cache the outputs with the current hash
            self.prev_outs_cache[curr_hash] = outs

            # Send final chunk
            final_response = {
                "id": f"chatcmpl-{time.time()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield "data: " + json.dumps(final_response) + "\n\n"


        return StreamingResponse(stream_results(), media_type="text/event-stream")


    def generate(self, request):
        raise NotImplementedError