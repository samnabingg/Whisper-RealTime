import whisper
import time
import torch
import numpy as np
import asyncio
import re
from scipy.signal import resample_poly
from math import gcd
from audio_streaming import audio_queue, stream, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE

model = whisper.load_model("base")

SILENCE_THRESHOLD = 0.01
BUFFER_SECONDS = 5

# Log file to store all transcriptions
LOG_FILE = "transcription_log.txt"


def resample_to_16k(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def is_silence(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < threshold


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def is_hallucination(text: str) -> bool:
    """Detect repeated word/phrase loops and known bad outputs."""
    # Known single-word hallucinations
    KNOWN_HALLUCINATIONS = {"you", "thank you", "bye", "bye.", "you.", "thank you.", ""}
    if text.lower().strip() in KNOWN_HALLUCINATIONS:
        return True

    words = text.strip().split()
    if len(words) >= 6:
        # If more than 60% of words are the same token, it's a loop
        most_common = max(set(words), key=words.count)
        if words.count(most_common) / len(words) > 0.6:
            return True

    return False


def log_transcription(text: str):
    """Append transcription with timestamp to log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")


async def transcribe_worker():
   
    buffer = []
    silent_chunks = 0

    while True:
        if not audio_queue.empty():
            chunk = audio_queue.get().flatten()

            if is_silence(chunk):
                silent_chunks += 1
                if silent_chunks > 3:
                    buffer = []  # Clear stale audio after 3s of silence
                continue
            else:
                silent_chunks = 0
                buffer.append(chunk)

            if len(buffer) >= BUFFER_SECONDS:
                audio_data = np.concatenate(buffer, axis=0)

                # Resample → normalize → clip
                audio_data = resample_to_16k(audio_data, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
                audio_data = normalize_audio(audio_data)
                audio_data = np.clip(audio_data, -1.0, 1.0)

                audio_padded = whisper.pad_or_trim(audio_data)
                mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)

                options = whisper.DecodingOptions(
                    language=None,
                    fp16=torch.cuda.is_available(),
                    without_timestamps=True,
                    suppress_tokens=[-1],
                )

                start = time.time()
                result = whisper.decode(model, mel, options)
                latency = time.time() - start

                text = result.text.strip()

                if is_hallucination(text):
                    print(f"[Skipped hallucination]: '{text[:60]}...' " if len(text) > 60 else f"[Skipped hallucination]: '{text}'")
                else:
                    print(f"{text} - {latency:.2f}s")
                    log_transcription(text)

                #Clear buffer fully — no sliding window overlap
                buffer = []

        await asyncio.sleep(0.01)


async def main():
    print(f"Listening at {DEVICE_SAMPLE_RATE}Hz → resampling to {WHISPER_SAMPLE_RATE}Hz")
    print(f"Logging transcriptions to: {LOG_FILE}\n")
    with stream:
        await transcribe_worker()


if __name__ == "__main__":
    asyncio.run(main())