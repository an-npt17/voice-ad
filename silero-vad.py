import os
import threading
import time
from typing import Callable, Optional

import numpy as np

# import serial
import sounddevice as sd
import torch

# from gpiozero import LED
from scipy import signal

from pysilero_vad import SileroVoiceActivityDetector


class SileroDenoiser:
    """Silero denoise model wrapper for real-time audio enhancement."""

    def __init__(self, model_name: str = "small_fast", device: str = "cpu"):
        """
        Initialize the Silero denoise model.

        Args:
            model_name: Model variant ('small_fast', 'large_fast', 'small_slow')
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model_name = model_name

        # Load the denoise model
        try:
            self.model, self.samples, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_denoise",
                name=model_name,
                device=self.device,
                trust_repo=True,
            )
            self.model.eval()
            torch.set_grad_enabled(False)
            self.read_audio, self.save_audio, self.denoise_func = self.utils
            print(f"Silero denoise model '{model_name}' loaded successfully")
        except Exception as e:
            print(f"Failed to load Silero denoise model: {e}")
            self.model = None

    def denoise_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denoise audio tensor in real-time.

        Args:
            audio_tensor: Input audio tensor [1, samples] or [samples]

        Returns:
            Denoised audio tensor
        """
        if self.model is None:
            return audio_tensor

        try:
            # Ensure correct dimensions [1, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 2 and audio_tensor.shape[0] != 1:
                audio_tensor = audio_tensor.T

            # Move to device and denoise
            audio_tensor = audio_tensor.to(self.device)

            with torch.no_grad():
                denoised = self.model(audio_tensor)

            return denoised.squeeze(0)  # Return [samples]

        except Exception as e:
            print(f"Denoising failed: {e}")
            return audio_tensor.squeeze(0) if audio_tensor.dim() > 1 else audio_tensor


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
    """Real-time voice activity detection using Silero VAD with sounddevice and optional denoising."""

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
        # New denoise parameters
        enable_denoise: bool = True,
        denoise_model: str = "small_fast",
        denoise_device: str = "cpu",
        denoise_chunk_size: int = 8000,  # Process in chunks for real-time performance
    ):
        """
        Initialize the real-time Silero VAD with optional denoising using sounddevice.

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
            enable_denoise: Whether to enable Silero denoising preprocessing
            denoise_model: Silero denoise model variant ('small_fast', 'large_fast', 'small_slow')
            denoise_device: Device for denoise model ('cpu' or 'cuda')
            denoise_chunk_size: Chunk size for denoising processing
        """
        self.vad = SileroVad(threshold=threshold, trigger_level=trigger_level)
        self.channels = channels
        self.samplerate = samplerate  # Target sample rate for VAD (16kHz)
        self.input_samplerate = (
            input_samplerate or samplerate
        )  # Input device sample rate
        self.device = device
        self.on_speech_detected = on_speech_detected
        self.save_detections = save_detections
        self.save_dir = save_dir
        self.verbose = verbose

        # Initialize denoiser if enabled
        self.enable_denoise = enable_denoise
        self.denoise_chunk_size = denoise_chunk_size
        if self.enable_denoise:
            self.denoiser = SileroDenoiser(
                model_name=denoise_model, device=denoise_device
            )
            if self.verbose:
                print(f"Denoise preprocessing enabled with model: {denoise_model}")
        else:
            self.denoiser = None
            if self.verbose:
                print("Denoise preprocessing disabled")

        # Setup resampling if needed
        self.needs_resampling = self.input_samplerate != self.samplerate
        if self.needs_resampling:
            if self.verbose:
                print(
                    f"Will resample from {self.input_samplerate}Hz to {self.samplerate}Hz"
                )

        # self.led = LED(17)
        # self.led.on()  # Turn on LED initially
        # time.sleep(3)  # Keep LED on for 3 seconds
        # self.led.off()
        #
        # self.ser = serial.Serial("/dev/serial0")
        # self.ser.baudrate = 9600
        self.last_message_time = 0  # Track the last time a message was sent

        detector_chunk_bytes = self.vad.detector.chunk_bytes()
        detector_chunk_samples = (
            detector_chunk_bytes // 2
        )  # 16-bit samples = 2 bytes per sample

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

        # Use target sample rate for buffer calculations
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

    def _resample_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Resample audio from input sample rate to target sample rate."""
        if not self.needs_resampling:
            return audio_data

        target_length = int(len(audio_data) * self.samplerate / self.input_samplerate)
        resampled = signal.resample(audio_data, target_length)
        return np.array(resampled, np.int16)

    def _denoise_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply denoising to audio chunk if enabled."""
        if (
            not self.enable_denoise
            or self.denoiser is None
            or self.denoiser.model is None
        ):
            return audio_data

        try:
            # Convert to float32 and normalize for denoising
            audio_float = audio_data.astype(np.float32) / 32767.0

            # Process in chunks for better real-time performance
            if len(audio_float) > self.denoise_chunk_size:
                denoised_chunks = []
                for i in range(0, len(audio_float), self.denoise_chunk_size):
                    chunk = audio_float[i : i + self.denoise_chunk_size]

                    # Pad chunk if too small
                    if len(chunk) < self.denoise_chunk_size:
                        chunk = np.pad(chunk, (0, self.denoise_chunk_size - len(chunk)))

                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
                    denoised_chunk = self.denoiser.denoise_audio(chunk_tensor)
                    denoised_chunks.append(denoised_chunk.cpu().numpy())

                denoised = np.concatenate(denoised_chunks)[: len(audio_float)]
            else:
                # Process small chunks directly
                chunk_tensor = torch.tensor(audio_float, dtype=torch.float32)
                denoised = self.denoiser.denoise_audio(chunk_tensor).cpu().numpy()

            # Convert back to int16
            denoised = np.clip(denoised * 32767.0, -32768, 32767)
            return denoised.astype(np.int16)

        except Exception as e:
            if self.verbose:
                print(f"Denoising failed, using original audio: {e}")
            return audio_data

    def _audio_callback(self, indata, frames, time, status):
        """sounddevice callback for streaming audio processing."""
        if status:
            if self.verbose:
                print(f"sounddevice status: {status}")

        audio_data = np.int16(indata[:, 0] * 32767)  # Only first channel if stereo

        # Apply resampling if needed
        if self.needs_resampling:
            audio_data = self._resample_audio(audio_data)

        # Apply denoising if enabled
        if self.enable_denoise:
            audio_data = self._denoise_audio_chunk(audio_data)

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
            # If we weren't already capturing speech, this is a new segment
            if not self._is_speech_active:
                if self.verbose:
                    prob = self.vad.last_prob
                    denoise_status = (
                        "with denoising" if self.enable_denoise else "without denoising"
                    )
                    print(
                        f"Speech detected ({prob:.3f}) - starting capture {denoise_status}"
                    )
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
                # self.led.off()

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

    def toggle_denoise(self, enabled: bool = False):
        """Toggle denoising on/off during runtime."""
        if not enabled:
            self.enable_denoise = not self.enable_denoise
        else:
            self.enable_denoise = enabled

        status = "enabled" if self.enable_denoise else "disabled"
        if self.verbose:
            print(f"Denoising {status}")

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

        # Start the input stream with input sample rate
        self._stream = sd.InputStream(
            samplerate=self.input_samplerate,
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

        denoise_info = (
            f" with {self.denoiser.model_name} denoising"
            if self.enable_denoise
            else " without denoising"
        )
        if self.verbose:
            print(f"Voice activity detection started{denoise_info}")

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

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            if self._is_speech_active:
                self._finalize_speech_segment()

        # self.led.off()
        if self.verbose:
            print("Voice activity detection stopped")


def demo():
    """Demo function to test the enhanced VAD model with denoising."""

    def on_speech(audio_bytes):
        duration_ms = (len(audio_bytes) / 2) / 16  # 16-bit samples at 16kHz
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

    print("\nSelect denoising mode:")
    print("[1] No denoising (original)")
    print("[2] Small Fast (recommended for real-time)")
    print("[3] Large Fast (better quality, more CPU)")
    print("[4] Small Slow (best quality, highest latency)")

    choice = input("Enter choice (1-4) or press Enter for default (2): ").strip()

    if choice == "1":
        enable_denoise = False
        denoise_model = None
    elif choice == "3":
        enable_denoise = True
        denoise_model = "large_fast"
    elif choice == "4":
        enable_denoise = True
        denoise_model = "small_slow"
    else:  # Default: choice == "2" or empty
        enable_denoise = True
        denoise_model = "small_fast"

    # Check if CUDA is available
    device_choice = "cuda" if torch.cuda.is_available() and enable_denoise else "cpu"
    if enable_denoise:
        print(f"Using device: {device_choice}")

    vad = SileroVADRealtimeSD(
        threshold=0.2,
        trigger_level=1,
        save_detections=False,
        on_speech_detected=on_speech,
        blocksize=2048,
        input_samplerate=48000,  # Input device runs at 48kHz
        samplerate=16000,  # VAD processes at 16kHz
        enable_denoise=enable_denoise,
        denoise_model=denoise_model if enable_denoise else "small_fast",
        denoise_device=device_choice,
        denoise_chunk_size=4000,  # Smaller chunks for lower latency
    )

    try:
        vad.start()

        print(f"\nListening for speech... Press Ctrl+C to stop.")
        if enable_denoise:
            print("Press 'd' + Enter to toggle denoising on/off")

        while True:
            try:
                # Non-blocking input check for denoising toggle
                import select
                import sys

                if select.select([sys.stdin], [], [], 0.1):
                    user_input = input().strip().lower()
                    if user_input == "d" and enable_denoise:
                        vad.toggle_denoise()
                else:
                    time.sleep(0.1)

            except:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping voice activity detection...")

    finally:
        # Clean up
        vad.stop()


if __name__ == "__main__":
    demo()
