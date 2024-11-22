from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
from matplotlib.animation import FuncAnimation

# Initialize the VAD model
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Parameters for audio capture
sample_rate = 16000
chunk_duration = 0.5
chunk_samples = int(chunk_duration * sample_rate)
speech_detected = False
audio_buffer = np.zeros(chunk_samples)
speech_probability = 0.0


# Process audio for VAD
def process_audio(chunk):
    global speech_detected
    timestamps = get_speech_timestamps(chunk, vad_model, sampling_rate=sample_rate)
    speech_detected = len(timestamps) > 0  # Set flag if speech is detected
    time = datetime.now()
    if speech_detected:
        print(f"{time}: Human voice detected!")


# Callback to read audio chunks in real-time
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)
    audio_data = np.mean(indata, axis=1)  # Convert to mono
    audio_buffer = audio_data.copy()
    process_audio(torch.tensor(audio_data, dtype=torch.float32))


speech_confidences = []

# Set up the plot
fig, ax = plt.subplots()
x_data = np.linspace(0, chunk_duration, chunk_samples)

(line,) = ax.plot(x_data, np.zeros(chunk_samples), lw=2)
ax.set_ylim(-1, 1)
ax.set_xlim(0, chunk_duration)


# Update function for animation
def update(frame):
    global speech_detected, audio_buffer
    # Update line data with current audio buffer
    line.set_ydata(audio_buffer)
    # Change line color based on speech detection
    line.set_color("red" if speech_detected else "blue")
    return (line,)


# Initialize the plot animation
with sd.InputStream(
    callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=chunk_samples
):
    ani = FuncAnimation(fig, update, blit=True, interval=chunk_duration * 1000)
    plt.title("Real-Time Voice Activity Detection")
    plt.show()
