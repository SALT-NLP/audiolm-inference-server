import torch
from transformers import AutoModel
from flask import Flask, request, jsonify
import numpy as np
import base64
from collections import defaultdict
import uuid
import time

app = Flask(__name__)

# Load the model
model = AutoModel.from_pretrained("WillHeld/DiVA-llama-3-v0-8b", trust_remote_code=True)

# Store client states
class ClientState:
    def __init__(self):
        self.outputs = None
        self.first_turn = True
        self.last_active = time.time()

client_states = {}

def cleanup_old_sessions(max_age=3600):  # remove sessions inactive for 1 hour
    current_time = time.time()
    to_remove = []
    for client_id, state in client_states.items():
        if current_time - state.last_active > max_age:
            to_remove.append(client_id)
    for client_id in to_remove:
        del client_states[client_id]

def process_stream(generator):
    """Helper function to collect all tokens from the stream"""
    full_response = ""
    try:
        while True:
            chunk, outputs = next(generator)
            full_response = chunk
    except StopIteration:
        return full_response, outputs

@app.route('/init_session', methods=['POST'])
def init_session():
    """Initialize a new client session"""
    client_id = str(uuid.uuid4())
    client_states[client_id] = ClientState()
    cleanup_old_sessions()  # Cleanup old sessions
    return jsonify({'client_id': client_id})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # get client ID and audio data from request
        data = request.json
        client_id = data.get('client_id')
        
        if client_id not in client_states:
            return jsonify({'error': 'Invalid session. Please initialize a new session.'}), 401
        
        state = client_states[client_id]
        state.last_active = time.time()
        
        audio_array = np.frombuffer(base64.b64decode(data['audio']), dtype=np.float32)
        
        with torch.no_grad():
            # generating a response
            generator = model.generate_stream(
                audio_array,
                text_prompt=("You are a voice assistant. Your interface with users will be voice. "
                           "You should use short and concise responses. You should tailor your response "
                           "style to speech using spelled out numbers and avoiding all formatting except "
                           "for standard punctuation.") if state.first_turn else None,
                do_sample=False,
                max_new_tokens=128,
                init_outputs=state.outputs if not state.first_turn else None,
                return_outputs=True
            )
            
            # processing the stream output from model
            response, state.outputs = process_stream(generator)
            state.first_turn = False
            
            return jsonify({'response': response})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)