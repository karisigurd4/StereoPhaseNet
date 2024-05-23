import pytest
import numpy as np
import librosa
from src.utils.data_processing import load_and_normalize_audio, generate_out_of_phase, frame_audio, frame_entire_audio

# Constants for test
TEST_AUDIO_PATH = 'tests/data/test_audio.wav'  # Ensure you have a test audio file at this path
FRAME_LENGTH = 16
HOP_LENGTH = 16

def test_load_and_normalize_audio():
    audio, sr = load_and_normalize_audio(TEST_AUDIO_PATH)
    
    assert audio.ndim == 2, "Audio is not stereo"
    assert audio.shape[0] == 2, "Audio does not have two channels"
    assert sr > 0, "Sample rate is not positive"
    assert np.max(np.abs(audio)) <= 1.0, "Audio is not normalized"

def test_generate_out_of_phase():
    audio, _ = load_and_normalize_audio(TEST_AUDIO_PATH)
    out_of_phase_audio = generate_out_of_phase(audio)
    
    assert out_of_phase_audio.shape == audio.shape, "Output shape is not the same as input"
    assert np.max(np.abs(out_of_phase_audio)) <= 1.0, "Output audio is not normalized"

def test_frame_audio():
    audio, _ = load_and_normalize_audio(TEST_AUDIO_PATH)
    frames = frame_audio(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    assert frames.ndim == 3, "Frames should be a 3D array"
    assert frames.shape[1] == 2, "Frames should have two channels"
    assert frames.shape[2] == FRAME_LENGTH, "Frames should have the correct frame length"

def test_frame_entire_audio():
    audio, _ = load_and_normalize_audio(TEST_AUDIO_PATH)
    frames = frame_entire_audio(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    assert frames.ndim == 3, "Frames should be a 3D array"
    assert frames.shape[1] == 2, "Frames should have two channels"
    assert frames.shape[2] == FRAME_LENGTH, "Frames should have the correct frame length"
    assert frames.shape[0] > 0, "There should be at least one frame"

if __name__ == "__main__":
    pytest.main([__file__])
