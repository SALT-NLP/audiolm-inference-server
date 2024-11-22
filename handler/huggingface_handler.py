import json
import time
from typing import AsyncGenerator, Generator
from fastapi.responses import StreamingResponse
import librosa
import numpy as np
import requests
from entity.entity import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    UsageInfo,
)
from handler.base_handler import BaseHandler
from datasets import Audio
from transformers import AutoModel
from transformers.pipelines.audio_utils import ffmpeg_read

from utils import random_uuid, get_file_from_any


class HuggingfaceHandler(BaseHandler):

    def __init__(self, model_name: str, sample_rate=16000):
        super().__init__(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.sample_rate = sample_rate

    def generate_stream(self, request):
        audio_infos_vllm = []
        for message in request.messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_infos_vllm.append(ele["audio_url"])
        inputs = {
            "messages": request.messages,
            "multi_modal_data": {
                "audio": [
                    librosa.load(get_file_from_any(a), sr=self.sample_rate)
                    for a in audio_infos_vllm
                ]
            },
        }

        y, sr = inputs["multi_modal_data"]["audio"][0]
        assert sr == self.sample_rate
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        results_generator = self.model.generate_stream(
            y,
            (
                "You are a helpful assistant The user is talking to you with their voice and you are responding with"
                " text."
            ),
            do_sample=request.temperature > 0.005,
            max_new_tokens=256,
        )

        # Streaming case
        def stream_results():
            prev_output = ""
            for text_output in results_generator:
                delta_text = text_output[len(prev_output):]
                finish_reason = None
                if delta_text == '':
                    finish_reason = 'stop'
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=delta_text),
                    logprobs=None,
                    finish_reason=finish_reason,
                )
                chunk = ChatCompletionStreamResponse(
                    id=request.request_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    choices=[choice_data],
                    model=self.model_name,
                )
                chunk.usage = UsageInfo(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                )

                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
                prev_output = text_output
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=prev_output),
                logprobs=None,
                finish_reason=finish_reason,
            )
            final_usage_chunk = ChatCompletionStreamResponse(
                id=request.request_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                choices=[choice_data],
                model=self.model_name,
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
            final_usage_data = final_usage_chunk.model_dump_json(
                exclude_unset=True, exclude_none=True
            )
            yield f"data: {final_usage_data}\n\n"

        return StreamingResponse(stream_results())

    def generate(self, request):
        raise NotImplementedError