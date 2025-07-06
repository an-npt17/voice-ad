import os
import threading
import time
import wave
from typing import Callable, List, Optional

import numpy as np
import serial
import sounddevice as sd
from gpiozero import LED

from pysilero_vad import SileroVoiceActivityDetector


class SileroVad:
    """Voice activity detection with silero VAD."""

    def __init__(self, threshold: float, trigger_level: int) -> None:
        self.detector = SileroVoiceActivityDetector()
        self.threshold = threshold
        self.trigger_level = trigger_level
        self._activation = 0
        self.last_prob = 0.0

    def __call__(self, audio_bytes: bytes | None) -> bool:
        if audio_bytes is None:
            # Reset
            self._activation = 0
            self.detector.reset()
            return False

        chunk_size = self.detector.chunk_bytes()
        if len(audio_bytes) < chunk_size:
            raise ValueError(f"Audio bytes must be at least {chunk_size} bytes")

        # Process chunks
        speech_probs = []
        for i in range(0, len(audio_bytes) - chunk_size + 1, chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            speech_probs.append(self.detector(chunk))

        # Use maximum probability
        max_prob = max(speech_probs)
        self.last_prob = max_prob
        if max_prob >= self.threshold:
            # Speech detected
            self._activation += 1
            if self._activation >= self.trigger_level:
                self._activation = 0
                return True
        else:
            # Silence detected
            self._activation = max(0, self._activation - 1)

        return False


class SileroVADRealtimeSD:
    """Real-time voice activity detection using Silero VAD with sounddevice."""

    def __init__(
        self,
        threshold: float = 0.15,
        trigger_level: int = 2,
        channels: int = 1,
        samplerate: int = 48000,
        model_samplerate: int = 16000,
        blocksize: Optional[int] = None,
        device: Optional[int] = None,
        on_speech_detected: Optional[Callable[[bytes], None]] = None,
        buffer_duration_ms: int = 500,
        min_silence_duration_ms: int = 500,
        save_detections: bool = False,
        save_dir: str = "speech_detections",
        verbose: bool = True,
    ):
        """
        Initialize the real-time Silero VAD using sounddevice.

        Args:
            threshold: Voice detection threshold between 0 and 1
            trigger_level: Number of consecutive detections needed to trigger speech
            channels: Number of audio channels (1 for mono, 2 for stereo)
            samplerate: Audio sampling rate in Hz (default: 16000)
            blocksize: Size of audio blocks in samples (if None, will use detector's chunk size)
            device: Input device index (None for default)
            on_speech_detected: Optional callback function when speech is detected
            buffer_duration_ms: Duration of audio buffer in milliseconds
            min_silence_duration_ms: Minimum silence duration to end a speech segment
            save_detections: Whether to save detected speech segments to WAV files
            save_dir: Directory to save speech detection files
            verbose: Whether to print status messages
        """
        self.vad = SileroVad(threshold=threshold, trigger_level=trigger_level)
        self.channels = channels
        self.samplerate = samplerate
        self.model_samplerate = model_samplerate
        self.need_resample = self.samplerate != self.model_samplerate

        self.device = device
        self.on_speech_detected = on_speech_detected
        self.save_detections = save_detections
        self.save_dir = save_dir
        self.verbose = verbose
        self.led = LED(17)
        self.led.on()  # Turn on LED initially
        time.sleep(3)  # Keep LED on for 3 seconds
        self.led.off()

        self.ser = serial.Serial("/dev/serial0")
        self.ser.baudrate = 9600
        self.last_message_time = 0  # Track the last time a message was sent

        detector_chunk_bytes = self.vad.detector.chunk_bytes()
        detector_chunk_samples = (
            detector_chunk_bytes // 2
        )  # 16-bit samples = 2 bytes per sample

        self.blocksize = blocksize if blocksize is not None else detector_chunk_samples

        if self.blocksize < detector_chunk_samples:
            self.blocksize = detector_chunk_samples

        samples_per_ms = self.samplerate / 1000
        self.buffer_size_samples = int(buffer_duration_ms * samples_per_ms)
        self.silence_threshold_samples = int(min_silence_duration_ms * samples_per_ms)

        if self.save_detections and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._is_running = False
        self._stream = None
        self._thread = None
        self._audio_buffer = np.array([], dtype=np.int16)
        self._speech_buffer = np.array([], dtype=np.int16)
        self._is_speech_active = False
        self._silence_counter = 0
        self._detection_count = 0

        # Lock for thread safety
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time, status):
        """sounddevice callback for streaming audio processing."""
        if status:
            if self.verbose:
                print(f"sounddevice status: {status}")

        # Extract audio data from current frame and convert to int16
        audio_data = np.int16(indata[:, 0] * 32767)  # Only first channel if stereo

        with self._lock:
            # Add to buffer
            self._audio_buffer = np.append(self._audio_buffer, audio_data)

            # Keep buffer at reasonable size
            if len(self._audio_buffer) > self.buffer_size_samples:
                # Process the buffer
                self._process_buffer()
                # Keep only the latest part of the buffer
                self._audio_buffer = self._audio_buffer[-self.blocksize :]

    def _process_buffer(self):
        """Process the audio buffer to detect voice activity."""
        # Make sure we have enough data
        if len(self._audio_buffer) < self.blocksize:
            return

        # Get the required chunk size in bytes (should be 1024)
        required_chunk_bytes = self.vad.detector.chunk_bytes()

        # Convert numpy array to bytes for VAD
        audio_bytes = self._audio_buffer.tobytes()

        if self.need_resample:
            import samplerate

            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            # Calculate number of output samples to match required chunk size
            required_samples = required_chunk_bytes // 2  # 2 bytes per int16 sample
            # Compute resample ratio
            resample_ratio = required_samples / len(audio_np)
            resampled = samplerate.resample(
                audio_np.astype(np.float32), resample_ratio, "sinc_best"
            )
            audio_bytes = np.asarray(resampled, dtype=np.int16).tobytes()

        is_speech = self.vad(audio_bytes)

        if is_speech:
            # If we weren't already capturing speech, this is a new segment
            if not self._is_speech_active:
                if self.verbose:
                    print("Speech detected - starting capture")
                self._is_speech_active = True
                self._speech_buffer = np.array([], dtype=np.int16)
                # Include some audio before the detection point (from buffer)
                self._speech_buffer = np.append(self._speech_buffer, self._audio_buffer)
                self.led.on()
                current_time = time.time()
                if current_time - self.last_message_time >= 10:
                    self.ser.write(b"{6}\n")
                    self.last_message_time = current_time

            else:
                # Continue adding to existing speech segment
                new_data = self._audio_buffer[-self.blocksize :]
                self._speech_buffer = np.append(self._speech_buffer, new_data)

            # Reset silence counter
            self._silence_counter = 0
        elif self._is_speech_active:
            # We're in an active speech segment but no speech detected
            # Add audio to speech buffer anyway in case speech restarts
            new_data = self._audio_buffer[-self.blocksize :]
            self._speech_buffer = np.append(self._speech_buffer, new_data)

            # Increment silence counter
            self._silence_counter += len(new_data)

            # If silence is long enough, end the speech segment
            if self._silence_counter >= self.silence_threshold_samples:
                if self.verbose:
                    print("Silence detected - ending speech capture")
                self._finalize_speech_segment()
                self.led.off()

    def _finalize_speech_segment(self):
        """Process the completed speech segment."""
        # Reset state
        self._is_speech_active = False
        self._silence_counter = 0

        # Convert numpy array to bytes
        speech_bytes = self._speech_buffer.tobytes()

        # Save the speech segment if requested
        if self.save_detections:
            filename = os.path.join(
                self.save_dir,
                f"speech_{time.strftime('%Y%m%d-%H%M%S')}_{self._detection_count}.wav",
            )
            self._save_audio(speech_bytes, filename)
            self._detection_count += 1

        # Call the callback if provided
        if self.on_speech_detected:
            self.on_speech_detected(speech_bytes)

        # Clear the speech buffer
        self._speech_buffer = np.array([], dtype=np.int16)

    def _save_audio(self, audio_bytes: bytes, filename: str):
        """Save audio bytes to a WAV file."""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_bytes)
        if self.verbose:
            print(f"Saved speech segment to {filename}")

    def _monitoring_thread(self):
        """Thread function for continuous monitoring."""
        try:
            while self._is_running:
                time.sleep(0.01)
                with self._lock:
                    # Process any data in the buffer
                    if len(self._audio_buffer) >= self.blocksize:
                        self._process_buffer()

                # Check if we need to finalize speech when stopping
                if not self._is_running and self._is_speech_active:
                    with self._lock:
                        self._finalize_speech_segment()
        except Exception as e:
            if self.verbose:
                print(f"Error in monitoring thread: {e}")

    def list_devices(self):
        """List available audio devices."""
        print(sd.query_devices())

    def start(self):
        """Start real-time voice activity detection."""
        if self._is_running:
            if self.verbose:
                print("Already running")
            return

        # Reset state
        with self._lock:
            self._is_running = True
            self._audio_buffer = np.array([], dtype=np.int16)
            self._speech_buffer = np.array([], dtype=np.int16)
            self._is_speech_active = False
            self._silence_counter = 0

            # Reset VAD
            self.vad(None)

            # Start the input stream
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
            latency="high",  # Use higher latency for more stable buffering
        )
        self._stream.start()

        # Start monitoring thread
        self._thread = threading.Thread(target=self._monitoring_thread)
        self._thread.daemon = True
        self._thread.start()

        if self.verbose:
            print("Voice activity detection started")

    def stop(self):
        """Stop real-time voice activity detection."""
        if not self._is_running:
            if self.verbose:
                print("Not running")
            return
        self.ser.close()

        # Set flag to stop
        self._is_running = False

        # Stop the stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Finalize any active speech segment
        with self._lock:
            if self._is_speech_active:
                self._finalize_speech_segment()

        self.led.off()
        if self.verbose:
            print("Voice activity detection stopped")


def demo():
    """Demo function to test the model class."""

    def on_speech(audio_bytes):
        duration_ms = (len(audio_bytes) / 2) / 16  # 16-bit samples at 16kHz
        print(f"Speech detected: {duration_ms:.2f}ms ({len(audio_bytes)} bytes)")

    devices = sd.query_devices()
    input_devices = []

    # List audio devices for debugging
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            print(f"[{i}] {device['name']}")
            input_devices.append(i)

    if not input_devices:
        print("No input devices found!")
        return

    # Create the VAD with selected device
    vad = SileroVADRealtimeSD(
        save_detections=False,
        on_speech_detected=on_speech,
    )

    try:
        vad.start()

        # Keep the program running
        print("\nListening for speech... Press Ctrl+C to stop.")
        while True:
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping voice activity detection...")

    finally:
        # Clean up
        vad.stop()


if __name__ == "__main__":
    demo()
