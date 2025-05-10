import torch
import numpy as np
import sounddevice as sd
from collections import deque
from vad import SileroVAD  # from pysilero-vad

class RealTimeVAD:
    def __init__(self, 
                 sample_rate=16000,
                 device="cpu",
                 threshold=0.5,
                 window_size=512,
                 speech_pad_ms=150,
                 trigger_level=0.6):
        self.sample_rate = sample_rate
        self.device = device
        self.threshold = threshold
        self.window_size = window_size
        self.speech_pad_ms = speech_pad_ms
        self.trigger_level = trigger_level

        self.vad = SileroVAD(sample_rate=self.sample_rate, 
                             threshold=self.threshold,
                             trigger_level=self.trigger_level,
                             speech_pad_ms=self.speech_pad_ms,
                             min_speech_duration_ms=250,
                             window_size_samples=self.window_size)

        self.buffer = deque()
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        if status:
            print("Stream error:", status)
        audio = indata[:, 0]  # Mono
        self.buffer.append(torch.from_numpy(audio.copy()).float())

    def start_stream(self):
        print("Starting audio stream...")
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype='float32',
            callback=self.audio_callback,
            blocksize=self.window_size,
        )
        self.stream.start()

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Stream stopped.")

    def run(self):
        try:
            self.start_stream()
            print("Listening... Press Ctrl+C to stop.")
            while True:
                if len(self.buffer) > 0:
                    audio_chunk = self.buffer.popleft()
                    is_speech = self.vad(audio_chunk)
                    if is_speech:
                        print("🔊 Speech detected!")
        except KeyboardInterrupt:
            self.stop_stream()
            print("Stopped by user.")

