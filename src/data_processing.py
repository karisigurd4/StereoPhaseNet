import librosa
import numpy as np

def load_and_normalize_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=False)
    return y, sr

def frame_entire_audio(audio, frame_length, hop_length):
    num_channels, num_samples = audio.shape
    num_frames = 1 + (num_samples - frame_length) // hop_length

    # Initialize the frames array
    frames = np.zeros((num_frames, num_channels, frame_length))

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[:, start:end]

        frames[i, :, :] = frame

    return frames

def apply_fft(audio_frames):
    return np.fft.rfft(audio_frames, axis=-1)