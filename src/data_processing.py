import librosa
import numpy as np

def load_and_normalize_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=False)
    y = librosa.util.normalize(y)
    return y, sr

def generate_out_of_phase(audio):
    # Ensure audio is stereo
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("Input audio must be stereo with shape (2, samples)")
    
    # Mid/Side transformation
    mid = (audio[0] + audio[1]) / 2.0
    side = (audio[0] - audio[1]) / 2.0
    
    # Normalize the side channel
    max_val = np.max(np.abs(side))
    if max_val > 0:
        side = side / max_val
    
    # Return the Side component as both channels in stereo format
    return np.array([side, side])

def frame_audio(audio, frame_length=16, hop_length=16):
    # Ensure audio is stereo
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("Input audio must be stereo with shape (2, samples)")
    
    # Frame each channel separately
    frames_left = librosa.util.frame(audio[0], frame_length=frame_length, hop_length=hop_length)
    frames_right = librosa.util.frame(audio[1], frame_length=frame_length, hop_length=hop_length)
    
    # Stack the frames along the new axis (num_frames, 2, frame_length)
    frames = np.stack((frames_left, frames_right), axis=1)
    
    # Ensure all frames are of the same length
    num_frames = frames.shape[2]
    if num_frames > frame_length:
        frames = frames[:, :, :frame_length]
    elif num_frames < frame_length:
        padding = np.zeros((frames.shape[0], frames.shape[1], frame_length - num_frames))
        frames = np.concatenate((frames, padding), axis=2)

    return frames  # Shape: (num_frames, 2, frame_length)

def frame_entire_audio(audio, frame_length=16, hop_length=16):
    frames = []
    for i in range(0, audio.shape[1] - frame_length + 1, hop_length):
        frame = audio[:, i:i + frame_length]
        frames.append(frame)
    frames = np.stack(frames, axis=0)
    return frames