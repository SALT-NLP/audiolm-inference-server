import pyaudio
import numpy as np
import requests
import base64
import signal
import sys

SERVER_URL = "http://172.24.67.78:5000"

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def init_session():
    """
    initialize a session with the server
    """
    try:
        response = requests.post(f"{SERVER_URL}/init_session")
        if response.status_code == 200:
            return response.json()['client_id']
        else:
            raise Exception("Failed to initialize session")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error during session initialization: {e}")

def record_audio(duration=3, sample_rate=16000):
    """
    record 3s long audio for input
    """
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    
    print(f"recording for {duration} seconds...")
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    for i in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("recording done")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
    return audio_data

def preprocess_audio(audio_data):
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    mask = np.abs(audio_data) > 0.01
    if np.any(mask):
        start = np.argmax(mask)
        end = len(audio_data) - np.argmax(mask[::-1])
        audio_data = audio_data[start:end]
    
    return audio_data

def send_audio_to_server(audio_data, client_id):
    audio_bytes = audio_data.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    
    try:
        response = requests.post(
            f"{SERVER_URL}/process_audio",
            json={
                'audio': audio_b64,
                'client_id': client_id
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Server error: {response.json().get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")

def main():
    print("starting convo:")
    
    try:
        print("intializing session...")
        client_id = init_session()
        print("init done")
    except Exception as e:
        print(f"Init ERROR: {e}")
        return
    
    while True:
        try:
            print("\npress end to start speaking...")
            input()
            
            # record and preproc audio
            audio_data = record_audio()
            audio_data = preprocess_audio(audio_data)
            
            # send to server and get response
            print("sending to server...")
            response = send_audio_to_server(audio_data, client_id)
            print("\nAssistant Response:", response)
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            print("\nMoving on...")
            continue

if __name__ == "__main__":
    main()