import base64
from openai import OpenAI
import requests
from typing import List


def encode_audio_base64_from_url(url_or_path: str) -> str:
    """Encode an audio retrieved from a remote url to base64 format."""
    if url_or_path.startswith("http"):
        with requests.get(url_or_path) as response:
            response.raise_for_status()
            result = base64.b64encode(response.content).decode("utf-8")
    else:
        with open(url_or_path, "rb") as wav_file:
            result = base64.b64encode(wav_file.read()).decode("utf-8")
    return result


def two_turn(audio_url_or_paths: List[str], model_name: str, stream=True):
    client = OpenAI(
        base_url="http://localhost:40021/v1",
        api_key="None",
    )
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. Respond conversationally to the speech provided.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "data:audio/wav;base64,"
                        + encode_audio_base64_from_url(audio_url_or_paths[0]),
                    },
                ],
            },
        ],
        model=model_name,
        max_tokens=64,
        stream=stream,
        temperature=0,
    )
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. Respond conversationally to the speech provided.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "data:audio/mp3;base64,"
                        + encode_audio_base64_from_url(audio_url_or_paths[0]),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Stanford University is located in Palo Alto, California.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "data:audio/mp3;base64,"
                        + encode_audio_base64_from_url(audio_url_or_paths[1]),
                    },
                ],
            },
        ],
        model=model_name,
        max_tokens=64,
        stream=stream,
        temperature=0,
    )
    response = ""
    for output in chat_completion_from_url:
        if len(output.choices) > 0:
            response += output.choices[0].delta.content
    assert (
        response == 'You said "What city is Stanford University located in?"'
    ), f"Actual Response was {response}"


def single_turn(audio_url_or_path: str, model_name: str, stream=True):
    client = OpenAI(
        base_url="http://localhost:40021/v1",
        api_key="None",
    )
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. Respond conversationally to the speech provided.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "data:audio/wav;base64,"
                        + encode_audio_base64_from_url(audio_url_or_path),
                    },
                ],
            },
        ],
        model=model_name,
        max_tokens=64,
        stream=stream,
        temperature=0,
    )
    response = ""
    for output in chat_completion_from_url:
        if len(output.choices) > 0:
            response += output.choices[0].delta.content
    assert (
        response == "Stanford University is located in Palo Alto, California."
    ), f"Actual Response was {response}"


if __name__ == "__main__":
    single_turn(
        audio_url_or_path="test/turn1.mp3", model_name="WillHeld/DiVA-llama-3-v0-8b"
    )
    two_turn(
        audio_url_or_paths=["test/turn1.mp3", "test/turn2.mp3"],
        model_name="WillHeld/DiVA-llama-3-v0-8b",
    )
