import datetime
import queue
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import torch
from matplotlib.animation import FuncAnimation


class SileroVADGraphWrapper:
    def __init__(
        self,
        threshold=0.5,
        sample_rate=16000,
        window_size_samples=512,
        display_graph=True,
        history_size=300,
        only_send_when_changed=True,
    ):
        """
        Initialize the Silero VAD wrapper with graphing capability

        Args:
            threshold: VAD threshold, higher values mean more sensitive
            sample_rate: Audio sample rate to use (in Hz)
            window_size_samples: Window size in samples - determines how many audio samples
                                 are processed at once (affects latency and sensitivity).
                                 Smaller values give faster response but might be less accurate.
            display_graph: Whether to display the real-time graph
            history_size: Number of points to keep in the graph history
            only_send_when_changed: Only print/send data when detection status changes
        """
        self.model, utils = torch.hub.load(  # pyright: ignore[reportGeneralTypeIssues]
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )

        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_size_samples = window_size_samples
        self.display_graph = display_graph
        self.history_size = history_size
        self.only_send_when_changed = only_send_when_changed

        self.window_duration_ms = (window_size_samples / sample_rate) * 1000

        print("Initializing Silero VAD with:")
        print(f"  - Threshold: {threshold}")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(
            f"  - Window size: {window_size_samples} samples ({self.window_duration_ms:.1f} ms)"
        )

        self.get_speech_timestamps = utils[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"  - Running on: {self.device}")

        self.model.eval()

        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.probabilities = deque(maxlen=history_size)
        self.times = deque(maxlen=history_size)
        self.detection_states = deque(maxlen=history_size)
        self.start_time = time.time()

        self.detection_log = []

        self.last_detection_state = False

        self.last_update_time = time.time()
        self.cumulative_time = 0.0

        if display_graph:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
            self.fig.suptitle("Silero VAD Real-time Sound Detection")

            self.ax1.set_ylim(0, 1)
            self.ax1.set_ylabel("Speech Probability")
            self.ax1.set_xlabel("Time (s)")
            self.ax1.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Threshold ({threshold})",
            )
            self.ax1.grid(True)
            self.ax1.legend()

            self.ax2.set_ylim(-0.5, 1.5)
            self.ax2.set_ylabel("Detection State")
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_yticks([0, 1])
            self.ax2.set_yticklabels(["False", "True"])
            self.ax2.grid(True)

            (self.line1,) = self.ax1.plot(
                [], [], "b-", lw=2, label="Speech Probability"
            )
            (self.line2,) = self.ax2.plot([], [], "g-", lw=2, drawstyle="steps-post")

            self.plot_window_size = 10.0

    def start_microphone_streaming(self):
        """Start capturing audio from microphone and display graph if enabled"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.window_size_samples,
            stream_callback=self._audio_callback,
        )

        print("\nListening for sounds... (Press Ctrl+C to stop)")
        print("Logging format: [TIMESTAMP] Detection_State (Speech_Probability)")
        print("---------------------------------------------------")

        try:
            processing_thread = threading.Thread(target=self._process_audio)
            processing_thread.daemon = True
            processing_thread.start()

            if self.display_graph:
                ani = FuncAnimation(
                    self.fig, self._update_graph, interval=100, blit=True
                )
                plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
                plt.show()
            else:
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop_event.set()
            stream.stop_stream()
            stream.close()
            p.terminate()

            if self.detection_log:
                self.save_detection_log()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def _process_audio(self):
        """Process audio chunks and detect speech"""
        self.start_time = time.time()
        self.last_update_time = self.start_time

        while not self.stop_event.is_set():
            try:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get()

                    current_time = time.time()
                    chunk_time = current_time - self.last_update_time
                    self.cumulative_time += chunk_time
                    self.last_update_time = current_time

                    audio_data = (
                        np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )

                    tensor = torch.from_numpy(audio_data).to(self.device)

                    speech_prob = self.model(tensor, self.sample_rate).item()

                    current_detection = speech_prob >= self.threshold

                    wall_clock_timestamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    )[:-3]
                    elapsed_time = time.time() - self.start_time

                    self.probabilities.append(speech_prob)
                    self.times.append(elapsed_time)
                    self.detection_states.append(1 if current_detection else 0)

                    if (
                        not self.only_send_when_changed
                        or current_detection != self.last_detection_state
                    ):
                        log_message = f"[{wall_clock_timestamp}] {current_detection} ({speech_prob:.4f})"
                        print(log_message)

                        self.detection_log.append(
                            {
                                "timestamp": wall_clock_timestamp,
                                "elapsed_time": elapsed_time,
                                "detection": current_detection,
                                "probability": speech_prob,
                            }
                        )

                    self.last_detection_state = current_detection
                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error processing audio: {e}")

    def _update_graph(self, frame):
        """Update the graph animation with proper time scaling"""
        if len(self.times) > 0:
            times_list = list(self.times)
            probs_list = list(self.probabilities)
            states_list = list(self.detection_states)

            self.line1.set_data(times_list, probs_list)

            self.line2.set_data(times_list, states_list)

            current_max_time = max(times_list)

            min_time = max(0, current_max_time - self.plot_window_size)
            max_time = current_max_time + 0.5

            if max_time > min_time:
                self.ax1.set_xlim(min_time, max_time)
                self.ax2.set_xlim(min_time, max_time)

                self.fig.suptitle(
                    f"Silero VAD Real-time Sound Detection - Elapsed Time: {current_max_time:.2f}s"
                )

        return (
            self.line1,
            self.line2,
        )

    def save_graph(self, filename="vad_detection_history.png"):
        """Save the current graph to a file"""
        if self.display_graph and len(self.times) > 0:
            plt.savefig(filename)
            print(f"Graph saved as {filename}")

    def save_detection_log(self, filename=None):
        """Save the detection log to a CSV file"""
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vad_detection_log_{timestamp}.csv"

        if self.detection_log:
            import csv

            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Timestamp", "Elapsed Time (s)", "Detection", "Probability"]
                )

                for entry in self.detection_log:
                    writer.writerow(
                        [
                            entry["timestamp"],
                            f"{entry['elapsed_time']:.3f}",
                            entry["detection"],
                            f"{entry['probability']:.4f}",
                        ]
                    )

            print(f"Detection log saved as {filename}")

    def get_current_data(self):
        """Return the current data for external use"""
        return {
            "times": list(self.times),
            "probabilities": list(self.probabilities),
            "detection_states": list(self.detection_states),
            "current_detection": self.last_detection_state,
            "detection_log": self.detection_log,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Silero VAD Real-time Sound Detection")
    parser.add_argument(
        "--file", type=str, help="Process audio file instead of microphone"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="VAD threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Audio sample rate (Hz)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=512,
        help="Window size in samples (smaller = more responsive but may be less accurate)",
    )
    parser.add_argument("--no_graph", action="store_true", help="Disable graph display")
    parser.add_argument(
        "--always_send",
        action="store_true",
        help="Always log detection state, not just on changes",
    )
    parser.add_argument(
        "--save_log", type=str, help="Save detection log to specified filename"
    )

    args = parser.parse_args()

    vad = SileroVADGraphWrapper(
        threshold=args.threshold,
        sample_rate=args.sample_rate,
        window_size_samples=args.window_size,
        display_graph=not args.no_graph,
        only_send_when_changed=not args.always_send,
    )

    vad.start_microphone_streaming()

    if args.save_log:
        vad.save_detection_log(args.save_log)
