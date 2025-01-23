from openai import OpenAI
import base64
import numpy as np
import pyaudio
import json
from typing import List

def record_audio(duration=3, sample_rate=16000):
    """Record audio using PyAudio"""
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    print(f"Recording for {duration} seconds...")
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    for _ in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Recording finished!")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return np.frombuffer(b''.join(frames), dtype=np.float32)

def preprocess_audio(audio_data):
    """Preprocess audio data"""
    audio_data = audio_data / np.max(np.abs(audio_data))
    mask = np.abs(audio_data) > 0.01
    if np.any(mask):
        start = np.argmax(mask)
        end = len(audio_data) - np.argmax(mask[::-1])
        audio_data = audio_data[start:end]
    return audio_data

def encode_audio_base64(audio_data):
    """Encode audio data to base64"""
    return base64.b64encode(audio_data.tobytes()).decode()

def main():
    client = OpenAI(
        base_url='http://172.24.67.130:40021/v1',
        api_key="None"
    )

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a voice assistant. Your interface with users will be voice. You should use short and concise responses. You should tailor your response style to speech using spelled out numbers and avoiding all formatting except for standard punctuation.",
                },
            ],
        }
    ]

    print("Starting conversation... Press Ctrl+C to exit")
    
    while True:
        try:
            print("\nPress Enter to start speaking...")
            input()
            
            # Record and process audio
            audio_data = record_audio()
            audio_data = preprocess_audio(audio_data)
            
            # Add audio message
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": f"data:audio/wav;base64,{encode_audio_base64(audio_data)}"
                    }
                ]
            })
            
            # Get response
            print("Processing...")
            try:
                response = ""
                stream = client.chat.completions.create(
                    messages=messages,
                    model="WillHeld/DiVA-llama-3-v0-8b",
                    max_tokens=128,
                    stream=True,
                    temperature=0,
                )
                
                # Process streaming response
                for output in stream:
                    if len(output.choices) > 0:                        
                        content = output.choices[0].delta.content
                        if content is not None:
                            print(content, end="", flush=True)
                            response += content
                print()
                
                # Add assistant's response to history
                if response:
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": response,
                            }
                        ]
                    })
                
            except Exception as e:
                print(f"\nError in stream processing: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Continuing...")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()