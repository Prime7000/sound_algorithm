import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask
from flask import render_template,jsonify,redirect,request
from groq import Groq
import sounddevice as sd
import soundfile as sf
import queue
import threading
import numpy as np
import time


app = Flask(__name__)

recording_state = None

record = False
task_thread = None

global auto_recording
auto_recording = 'false'

global auto_transcription
auto_transcription = 'no transcription'

def automatic_detection_loop():
    # CONFIG
    THRESHOLD = 1.00
    SILENCE_DURATION = 3
    SAMPLE_RATE = 44100
    CHANNELS = 1
    BLOCKSIZE = 1024

    def perform_action(audio_data):
        filename = "mic.wav"
        sf.write(filename, np.concatenate(audio_data, axis=0), SAMPLE_RATE)
        print(f"Saved to {filename}")

        api_key = os.getenv('GROQ_API_KEY')
        client = Groq(api_key=api_key)
        
        with open(filename, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                response_format="verbose_json",
                language="en",
                timestamp_granularities=["segment", "word"]
            )

        print(resp.text)
        global auto_transcription
        auto_transcription = resp.text
        return resp.text

    state = "listening"
    last_sound_time = 0
    recorded_data = []

    print("🎧 Listening continuously... (Auto Mode ON)")

    with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE) as stream:
        try:
            while recording_state == "automatic_mode":
                data, _ = stream.read(BLOCKSIZE)
                volume = np.linalg.norm(data)
                now = time.time()

                if state == "listening":
                    if volume > THRESHOLD:
                        print("🎤 Sound detected above threshold...")
                        state = "recording"
                        last_sound_time = now
                        globals()['auto_recording'] = 'true'
                        recorded_data.append(data)

                elif state == "recording":
                    recorded_data.append(data)
                    if volume > THRESHOLD:
                        last_sound_time = now
                    elif now - last_sound_time > SILENCE_DURATION:
                        print("🎤 Silence detected, processing...")
                        perform_action(recorded_data)
                        recorded_data = []
                        state = "listening"
                        globals()['auto_recording'] = 'false'

                time.sleep(0.01)
        except Exception as e:
            print("Error in automatic loop:", e)


# recording task
def recording_voice():
    samplerate = 44100
    channels = 1
    filename = "mic.wav"
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    print("Recording... Press Ctrl+C to stop.")

    # Store chunks here
    recorded = []

    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32', callback=callback):
            while record == True:
                chunk = q.get()
                recorded.append(chunk)

    except KeyboardInterrupt:
        print("\nStopped recording.")

    # Concatenate and save
    audio = np.concatenate(recorded, axis=0)
    sf.write(filename, audio, samplerate)
    print(f"Saved to {filename}")



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/call_audio')
def call_audio():
    global record

    if recording_state == 'manual_mode':
        record = True
        task_thread = threading.Thread(target=recording_voice ,daemon=True).start()


        dataX = {
            'finished':'recording'
        }
        return jsonify(dataX)
    else:
        record = False
        dataX = {
            'finished':'automatic mode is on'
        }
        print('cannot record because auto is on', recording_state)
        return jsonify(dataX)

@app.route('/stop_recording')
def stop_recording():
    global record

    record = False
    return jsonify({'message': "stop recording"})
    

@app.route('/transcribe_audio')
def transcribe():
    if recording_state == 'manual_mode':
        api_key = os.getenv('GROQ_API_KEY')
        client = Groq(api_key=api_key)
        
        with open("mic.wav", "rb") as f:
            resp = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                response_format="verbose_json",
                language="en",
                timestamp_granularities=["segment", "word"]
            )

        print(resp.text)


        data = {
            'transcribe':resp.text,
            'finished':'true'
        }
        return jsonify(data)
    else:
        data ={
            'transcribe':auto_transcription
        }

        return jsonify(data)

@app.route('/state', methods=['POST'])
def state():
    global recording_state, task_thread

    if request.method == 'POST':
        state = request.form.get('transcribe_state')
        print(f'state --------------------- {state}')
        recording_state = state

        if recording_state == 'automatic_mode':
            print("Starting auto mode thread...")
            task_thread = threading.Thread(target=automatic_detection_loop, daemon=True)
            task_thread.start()

        data = {
            'state': 'successfully received'
        }
        return jsonify(data)


@app.route('/auto_record')
def auto_record():

    dataX = {
        'auto_record':auto_recording
    }
    print('auto recording status', auto_recording)
    print('recording state is ',recording_state)

    return jsonify(dataX)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
    