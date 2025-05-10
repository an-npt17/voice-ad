import sounddevice as sd
import numpy as np
import threading
import time
import wave
import os
from typing import Optional, Callable, List
from pysilero_vad import SileroVoiceActivityDetector


class SileroVad:
    """Voice activity detection with silero VAD."""
    def __init__(self, threshold: float, trigger_level: int) -> None:
        self.detector = SileroVoiceActivityDetector()
        self.threshold = threshold
        self.trigger_level = trigger_level
        self._activation = 0
    
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
        threshold: float = 0.5,
        trigger_level: int = 2,
        channels: int = 1,
        samplerate: int = 16000,
        blocksize: Optional[int] = None,
        device: Optional[int] = None,
        on_speech_detected: Optional[Callable[[bytes], None]] = None,
        buffer_duration_ms: int = 500,
        min_silence_duration_ms: int = 1000,
        save_detections: bool = False,
        save_dir: str = "speech_detections",
        verbose: bool = True
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
        self.device = device
        self.on_speech_detected = on_speech_detected
        self.save_detections = save_detections
        self.save_dir = save_dir
        self.verbose = verbose
        
        # Calculate chunk size based on detector's requirements
        detector_chunk_bytes = self.vad.detector.chunk_bytes()
        detector_chunk_samples = detector_chunk_bytes // 2  # 16-bit samples = 2 bytes per sample
        
        # Set blocksize (in samples)
        self.blocksize = blocksize if blocksize is not None else detector_chunk_samples
        
        # Make sure blocksize is at least the detector's required size
        if self.blocksize < detector_chunk_samples:
            self.blocksize = detector_chunk_samples
            
        # Calculate buffer sizes in samples
        samples_per_ms = self.samplerate / 1000
        self.buffer_size_samples = int(buffer_duration_ms * samples_per_ms)
        self.silence_threshold_samples = int(min_silence_duration_ms * samples_per_ms)
        
        # Create save directory if needed
        if self.save_detections and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # State variables
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
                self._audio_buffer = self._audio_buffer[-self.blocksize:]
    
    def _process_buffer(self):
        """Process the audio buffer to detect voice activity."""
        # Make sure we have enough data
        if len(self._audio_buffer) < self.blocksize:
            return
        
        # Convert numpy array to bytes for VAD
        audio_bytes = self._audio_buffer.tobytes()
        
        # Check for speech
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
            else:
                # Continue adding to existing speech segment
                new_data = self._audio_buffer[-self.blocksize:]
                self._speech_buffer = np.append(self._speech_buffer, new_data)
            
            # Reset silence counter
            self._silence_counter = 0
        elif self._is_speech_active:
            # We're in an active speech segment but no speech detected
            # Add audio to speech buffer anyway in case speech restarts
            new_data = self._audio_buffer[-self.blocksize:]
            self._speech_buffer = np.append(self._speech_buffer, new_data)
            
            # Increment silence counter
            self._silence_counter += len(new_data)
            
            # If silence is long enough, end the speech segment
            if self._silence_counter >= self.silence_threshold_samples:
                if self.verbose:
                    print("Silence detected - ending speech capture")
                self._finalize_speech_segment()
    
    def _finalize_speech_segment(self):
        """Process the completed speech segment."""
        # Reset state
        self._is_speech_active = False
        self._silence_counter = 0
        
        # Convert numpy array to bytes
        speech_bytes = self._speech_buffer.tobytes()
        
        # Save the speech segment if requested
        if self.save_detections:
            filename = os.path.join(self.save_dir, f"speech_{time.strftime('%Y%m%d-%H%M%S')}_{self._detection_count}.wav")
            self._save_audio(speech_bytes, filename)
            self._detection_count += 1
        
        # Call the callback if provided
        if self.on_speech_detected:
            self.on_speech_detected(speech_bytes)
        
        # Clear the speech buffer
        self._speech_buffer = np.array([], dtype=np.int16)
    
    def _save_audio(self, audio_bytes: bytes, filename: str):
        """Save audio bytes to a WAV file."""
        with wave.open(filename, 'wb') as wf:
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
                time.sleep(0.1)  # Check every 100ms
                
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
            dtype='float32',
            callback=self._audio_callback
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
        
        if self.verbose:
            print("Voice activity detection stopped")


def demo():
    """Demo function to test the SileroVADRealtimeSD class."""
    def on_speech(audio_bytes):
        duration_ms = (len(audio_bytes) / 2) / 16  # 16-bit samples at 16kHz
        print(f"Speech detected: {duration_ms:.2f}ms ({len(audio_bytes)} bytes)")
    
    # Create the VAD
    vad = SileroVADRealtimeSD(
        threshold=0.5,
        trigger_level=2,
        save_detections=False,
        on_speech_detected=on_speech
    )
    
    try:
        # List available devices
        print("Available audio devices:")
        vad.list_devices()
        
        # Start the VAD
        vad.start()
        
        # Keep the program running
        print("Listening for speech... Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping voice activity detection...")
    
    finally:
        # Clean up
        vad.stop()


if __name__ == "__main__":
    demo()
