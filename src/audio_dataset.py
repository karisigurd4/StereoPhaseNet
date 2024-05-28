from src.data_processing import load_and_normalize_audio, frame_entire_audio, apply_fft
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

def frames_to_fft_with_phase(audio_frames):
        # Compute the FFT of the input audio frames
        fft_frames = np.fft.rfft(audio_frames, axis=-1)

        # Extract magnitude and phase
        magnitude = np.abs(fft_frames)
        phase = np.angle(fft_frames)

        # Combine magnitude and phase into a new array with shape (num_frames, 4, frame_length)
        # Assuming audio_frames.shape is (num_frames, 2, frame_length)
        num_frames, num_channels, frame_length = audio_frames.shape
        half_frame_length = magnitude.shape[-1]  # rfft reduces the last dimension to half+1

        combined_frames = np.zeros((num_frames, num_channels * 2, half_frame_length))

        # Fill in the combined frames
        combined_frames[:, 0:num_channels, :] = magnitude
        combined_frames[:, num_channels:num_channels * 2, :] = phase

        return combined_frames

class AudioDataset(Dataset):
    def __init__(self, input_files, target_files, frame_length=512, hop_length=256):
        self.frames = []
        self.targets = []

        for input_file, target_file in zip(input_files, target_files):
            input_audio, sr = load_and_normalize_audio(input_file)
            target_audio, sr = load_and_normalize_audio(target_file)
            
            print (f"Input file: {input_file} - Target file: {target_file}")

            # Ensure audio is stereo
            if input_audio.ndim == 1:
                input_audio = np.stack([input_audio, input_audio])
            if target_audio.ndim == 1:
                target_audio = np.stack([target_audio, target_audio])
            
            # Frame the entire audio
            input_frames = frame_entire_audio(input_audio, frame_length=frame_length, hop_length=hop_length)
            target_frames = frame_entire_audio(target_audio, frame_length=frame_length, hop_length=hop_length)
            
            delta_frames = target_frames - input_frames

            self.frames.extend(input_frames)
            self.targets.extend(delta_frames)

        self.frames = np.array(self.frames)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.frames[idx], self.targets[idx])