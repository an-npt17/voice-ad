import torch

model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True, onnx=True
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
