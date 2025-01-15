from .huggingface_handler import HuggingfaceHandler


def get_handler(model_name: str):
    # if model_name in ['Qwen/Qwen2-Audio-7B-Instruct']:
    #    return VLLMHandler(model_name)
    return HuggingfaceHandler(model_name)
