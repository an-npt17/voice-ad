import os
import threading
import time
from typing import Callable, Optional

import numpy as np
import requests
import serial
import sounddevice as sd
from dotenv import load_dotenv
from gpiozero import LED
from scipy import signal

from pysilero_vad import SileroVoiceActivityDetector

load_dotenv()


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

        speech_probs = []
        for i in range(0, len(audio_bytes) - chunk_size + 1, chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            speech_probs.append(self.detector(chunk))

        max_prob = max(speech_probs)
        self.last_prob = max_prob
        if max_prob >= self.threshold:
            self._activation += 1
            if self._activation >= self.trigger_level:
                self._activation = 0
                return True
        else:
            self._activation = max(0, self._activation - 1)

        return False


class SileroVADRealtimeSD:
    """Real-time voice activity detection using Silero VAD with sounddevice."""

    def __init__(
        self,
        threshold: float = 0.2,
        trigger_level: int = 2,
        channels: int = 1,
        samplerate: int = 16000,
        input_samplerate: Optional[int] = None,
        blocksize: Optional[int] = None,
        device: Optional[int] = None,
        on_speech_detected: Optional[Callable[[bytes], None]] = None,
        buffer_duration_ms: int = 500,
        min_silence_duration_ms: int = 500,
        save_detections: bool = False,
        save_dir: str = "speech_detections",
        verbose: bool = True,
        thingspeak_api_key: Optional[str] = None,
        thingspeak_field: int = 1,
    ):
        """
        Initialize the real-time Silero VAD using sounddevice.

        Args:
            threshold: Voice detection threshold between 0 and 1
            trigger_level: Number of consecutive detections needed to trigger speech
            channels: Number of audio channels (1 for mono, 2 for stereo)
            samplerate: Target sampling rate for VAD processing (default: 16000)
            input_samplerate: Input device sampling rate (if None, uses samplerate)
            blocksize: Size of audio blocks in samples (if None, will use detector's chunk size)
            device: Input device index (None for default)
            on_speech_detected: Optional callback function when speech is detected
            buffer_duration_ms: Duration of audio buffer in milliseconds
            min_silence_duration_ms: Minimum silence duration to end a speech segment
            save_detections: Whether to save detected speech segments to WAV files
            save_dir: Directory to save speech detection files
            verbose: Whether to print status messages
            thingspeak_api_key: ThingSpeak Write API Key for sending detection counts
            thingspeak_field: ThingSpeak field number to send detection count (1-8)
        """
        self.vad = SileroVad(threshold=threshold, trigger_level=trigger_level)
        self.channels = channels
        self.samplerate = samplerate
        self.input_samplerate = input_samplerate or samplerate
        self.device = device
        self.on_speech_detected = on_speech_detected
        self.save_detections = save_detections
        self.save_dir = save_dir
        self.verbose = verbose

        # ThingSpeak configuration
        self.thingspeak_api_key = thingspeak_api_key
        self.thingspeak_field = thingspeak_field
        self.thingspeak_url = "https://api.thingspeak.com/update"
        self.minute_detection_count = 0
        self.last_thingspeak_update = time.time()
        self._thingspeak_lock = threading.Lock()

        # Setup resampling if needed
        self.needs_resampling = self.input_samplerate != self.samplerate
        if self.needs_resampling:
            if self.verbose:
                print(
                    f"Will resample from {self.input_samplerate}Hz to {self.samplerate}Hz"
                )

        # self.led = LED(17)
        # self.led.on()
        # time.sleep(3)
        # self.led.off()

        # self.ser = serial.Serial("/dev/serial0")
        # self.ser.baudrate = 9600
        self.last_message_time = 0

        detector_chunk_bytes = self.vad.detector.chunk_bytes()
        detector_chunk_samples = detector_chunk_bytes // 2

        if self.needs_resampling:
            self.blocksize = (
                blocksize
                if blocksize is not None
                else int(
                    detector_chunk_samples * self.input_samplerate / self.samplerate
                )
            )
        else:
            self.blocksize = (
                blocksize if blocksize is not None else detector_chunk_samples
            )

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
        self._thingspeak_thread = None
        self._audio_buffer = np.array([], dtype=np.int16)
        self._speech_buffer = np.array([], dtype=np.int16)
        self._is_speech_active = False
        self._silence_counter = 0
        self._detection_count = 0

        self._lock = threading.Lock()

    def _send_to_thingspeak(self, count: int):
        """Send detection count to ThingSpeak."""
        if not self.thingspeak_api_key:
            return

        try:
            payload = {
                "api_key": self.thingspeak_api_key,
                f"field{self.thingspeak_field}": count,
            }
            response = requests.post(self.thingspeak_url, data=payload, timeout=10)

            if response.status_code == 200:
                if self.verbose:
                    print(f"ThingSpeak: Sent {count} detections successfully")
            else:
                if self.verbose:
                    print(
                        f"ThingSpeak: Failed to send data (status {response.status_code})"
                    )
        except Exception as e:
            if self.verbose:
                print(f"ThingSpeak: Error sending data - {e}")

    def _thingspeak_monitoring_thread(self):
        """Thread function for periodic ThingSpeak updates."""
        try:
            while self._is_running:
                time.sleep(1)  # Check every second

                current_time = time.time()
                elapsed = current_time - self.last_thingspeak_update

                # Send update every 60 seconds (1 minute)
                if elapsed >= 60:
                    with self._thingspeak_lock:
                        count = self.minute_detection_count
                        self.minute_detection_count = 0
                        self.last_thingspeak_update = current_time

                    if self.verbose:
                        print(
                            f"ThingSpeak: Reporting {count} detections in last minute"
                        )

                    self._send_to_thingspeak(count)
        except Exception as e:
            if self.verbose:
                print(f"Error in ThingSpeak thread: {e}")

    def _resample_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Resample audio from input sample rate to target sample rate."""
        if not self.needs_resampling:
            return audio_data

        target_length = int(len(audio_data) * self.samplerate / self.input_samplerate)
        resampled = signal.resample(audio_data, target_length)
        return np.array(resampled, np.int16)

    def _audio_callback(self, indata, frames, time, status):
        """sounddevice callback for streaming audio processing."""
        if status:
            if self.verbose:
                print(f"sounddevice status: {status}")

        audio_data = np.int16(indata[:, 0] * 32767)

        if self.needs_resampling:
            audio_data = audio_data.astype(np.ndarray)
            audio_data = self._resample_audio(audio_data)

        with self._lock:
            self._audio_buffer = np.append(self._audio_buffer, audio_data)

            if len(self._audio_buffer) > self.buffer_size_samples:
                self._process_buffer()
                self._audio_buffer = self._audio_buffer[-self.blocksize :]

    def _process_buffer(self):
        """Process the audio buffer to detect voice activity."""
        if len(self._audio_buffer) < self.blocksize:
            return

        audio_bytes = self._audio_buffer.tobytes()

        is_speech = self.vad(audio_bytes)

        if is_speech:
            if not self._is_speech_active:
                if self.verbose:
                    print("Speech detected - starting capture")
                self._is_speech_active = True
                self._speech_buffer = np.array([], dtype=np.int16)
                self._speech_buffer = np.append(self._speech_buffer, self._audio_buffer)
                # self.led.on()
                current_time = time.time()
                if current_time - self.last_message_time >= 10:
                    # self.ser.write(b"{6}\n")
                    self.last_message_time = current_time

            else:
                new_data = self._audio_buffer[-self.blocksize :]
                self._speech_buffer = np.append(self._speech_buffer, new_data)

            self._silence_counter = 0
        elif self._is_speech_active:
            new_data = self._audio_buffer[-self.blocksize :]
            self._speech_buffer = np.append(self._speech_buffer, new_data)

            self._silence_counter += len(new_data)

            if self._silence_counter >= self.silence_threshold_samples:
                if self.verbose:
                    print("Silence detected - ending speech capture")
                self._finalize_speech_segment()

    def _finalize_speech_segment(self):
        """Process the completed speech segment."""
        self._is_speech_active = False
        self._silence_counter = 0

        speech_bytes = self._speech_buffer.tobytes()

        if self.save_detections:
            filename = os.path.join(
                self.save_dir,
                f"speech_{time.strftime('%Y%m%d-%H%M%S')}_{self._detection_count}.wav",
            )
            self._detection_count += 1

        # Increment the minute detection counter
        with self._thingspeak_lock:
            self.minute_detection_count += 1

        if self.on_speech_detected:
            self.on_speech_detected(speech_bytes)

        self._speech_buffer = np.array([], dtype=np.int16)

    def _monitoring_thread(self):
        """Thread function for continuous monitoring."""
        try:
            while self._is_running:
                time.sleep(0.01)
                with self._lock:
                    if len(self._audio_buffer) >= self.blocksize:
                        self._process_buffer()

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

        with self._lock:
            self._is_running = True
            self._audio_buffer = np.array([], dtype=np.int16)
            self._speech_buffer = np.array([], dtype=np.int16)
            self._is_speech_active = False
            self._silence_counter = 0

            self.vad(None)

        with self._thingspeak_lock:
            self.minute_detection_count = 0
            self.last_thingspeak_update = time.time()

        self._stream = sd.InputStream(
            samplerate=self.input_samplerate,
            blocksize=self.blocksize,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
            latency="high",
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._monitoring_thread)
        self._thread.daemon = True
        self._thread.start()

        # Start ThingSpeak monitoring thread
        if self.thingspeak_api_key:
            self._thingspeak_thread = threading.Thread(
                target=self._thingspeak_monitoring_thread
            )
            self._thingspeak_thread.daemon = True
            self._thingspeak_thread.start()
            if self.verbose:
                print("ThingSpeak monitoring started")

        if self.verbose:
            print("Voice activity detection started")

    def stop(self):
        """Stop real-time voice activity detection."""
        if not self._is_running:
            if self.verbose:
                print("Not running")
            return
        # self.ser.close()

        self._is_running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._thingspeak_thread:
            self._thingspeak_thread.join(timeout=2.0)
            self._thingspeak_thread = None

        with self._lock:
            if self._is_speech_active:
                self._finalize_speech_segment()

        # Send final count to ThingSpeak before stopping
        if self.thingspeak_api_key:
            with self._thingspeak_lock:
                if self.minute_detection_count > 0:
                    if self.verbose:
                        print(
                            f"ThingSpeak: Sending final count of {self.minute_detection_count}"
                        )
                    self._send_to_thingspeak(self.minute_detection_count)
                    self.minute_detection_count = 0

        # self.led.off()
        if self.verbose:
            print("Voice activity detection stopped")


def demo():
    """Demo function to test the model class."""

    def on_speech(audio_bytes):
        duration_ms = (len(audio_bytes) / 2) / 16
        print(f"Speech detected: {duration_ms:.2f}ms ({len(audio_bytes)} bytes)")

    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            print(f"[{i}] {device['name']}")
            input_devices.append(i)

    if not input_devices:
        print("No input devices found!")
        return

    # Replace with your actual ThingSpeak Write API Key
    THINGSPEAK_API_KEY = os.getenv("THINGSPEAK_API_KEY")

    vad = SileroVADRealtimeSD(
        threshold=0.2,
        trigger_level=1,
        save_detections=False,
        on_speech_detected=on_speech,
        blocksize=2048,
        input_samplerate=16000,
        samplerate=16000,
        thingspeak_api_key=THINGSPEAK_API_KEY,
        thingspeak_field=1,
    )

    try:
        vad.start()

        print("\nListening for speech... Press Ctrl+C to stop.")
        print("Detection counts will be sent to ThingSpeak every minute.")
        while True:
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping voice activity detection...")

    finally:
        vad.stop()


if __name__ == "__main__":
    demo()
