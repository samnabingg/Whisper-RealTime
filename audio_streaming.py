import sounddevice as sd
import numpy as np
import queue

print(sd.query_devices())

# Use device 9 (WASAPI) but query its native sample rate
DEVICE_INDEX = 9
device_info = sd.query_devices(DEVICE_INDEX, 'input')
DEVICE_SAMPLE_RATE = int(device_info['default_samplerate'])  # likely 44100 or 48000
WHISPER_SAMPLE_RATE = 16000  # Whisper always needs 16kHz

print(f"Device sample rate: {DEVICE_SAMPLE_RATE}")

CHUNK_DURATION = 1.0  # seconds
CHUNK_SIZE = int(DEVICE_SAMPLE_RATE * CHUNK_DURATION)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"[Audio status]: {status}")
    audio_queue.put(indata.copy())

stream = sd.InputStream(
    samplerate=DEVICE_SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    dtype='float32',
    blocksize=CHUNK_SIZE,
    device=DEVICE_INDEX
)