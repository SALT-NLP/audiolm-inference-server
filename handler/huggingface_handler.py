import time
import xxhash
from fastapi.responses import StreamingResponse
import librosa
import numpy as np
import torch
from entity.entity import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    UsageInfo,
)
from handler.base_handler import BaseHandler
from transformers import AutoModel

from utils import get_either, get_file_from_any


class HuggingfaceHandler(BaseHandler):
    def __init__(self, model_name: str, sample_rate=16000):
        super().__init__(model_name)
        self.prev_outs_cache = {}
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, device_map="balanced_low_0"
        )
        self.sample_rate = sample_rate

    @torch.no_grad()
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

        prev_outs = None
        if len(inputs["multi_modal_data"]["audio"]) > 1:
            y, sr = inputs["multi_modal_data"]["audio"][-2]
            prev_hash = xxhash.xxh32(bytes(y)).hexdigest()
            prev_outs = self.prev_outs_cache

        y, sr = inputs["multi_modal_data"]["audio"][-1]
        curr_hash = xxhash.xxh32(bytes(y)).hexdigest()
        print("Curr Hash")
        print(curr_hash)
        assert sr == self.sample_rate
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        results_generator = self.model.generate_stream(
            y,
            (
                "You are a voice assistant. Your interface with users will be voice. You should use short and concise responses. You should tailor your response style to speech using spelled out numbers and avoiding all formatting except for standard punctuation."
                if prev_outs == None
                else None
            ),
            do_sample=request.temperature > 0.005,
            max_new_tokens=get_either(
                [request.max_completion_tokens, request.max_tokens]
            ),
            init_outputs=prev_outs,
            return_outputs=True,
        )

        # Streaming case
        @torch.no_grad()
        def stream_results():
            prev_output = ""
            for text_output, outs in results_generator:
                delta_text = text_output[len(prev_output) :]
                finish_reason = None
                if delta_text == "":
                    finish_reason = "stop"
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
            self.prev_outs_cache = outs
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=""),
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
