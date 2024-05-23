import pytest
import torch
from torch.utils.data import DataLoader
from src.audio_transformer import AudioTransformer
from src.train import AudioDataset, calculate_phase_coherence, train_model
from src.data_processing import load_and_normalize_audio, generate_out_of_phase, frame_entire_audio
import numpy as np
import os

# Constants for test
TEST_AUDIO_PATH = 'tests/data/test_audio.wav'  # Ensure you have a test audio file at this path
FRAME_LENGTH = 512
HOP_LENGTH = 256
BATCH_SIZE = 8

def test_audio_dataset():
    audio_files = [TEST_AUDIO_PATH]
    dataset = AudioDataset(audio_files, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    assert len(dataset) > 0, "Dataset should not be empty"
    
    for i in range(len(dataset)):
        frames, targets = dataset[i]
        assert frames.shape == (2, FRAME_LENGTH), "Frame shape is incorrect"
        assert targets.shape == (2, FRAME_LENGTH), "Target shape is incorrect"

def test_data_loader():
    audio_files = [TEST_AUDIO_PATH]
    dataset = AudioDataset(audio_files, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for batch_idx, (frames, targets) in enumerate(dataloader):
        assert frames.shape[0] <= BATCH_SIZE, "Batch size is incorrect"
        assert frames.shape[1:] == (2, FRAME_LENGTH), "Frame shape in batch is incorrect"
        assert targets.shape[1:] == (2, FRAME_LENGTH), "Target shape in batch is incorrect"
        break  # Test only the first batch

def test_calculate_phase_coherence():
    audio, _ = load_and_normalize_audio(TEST_AUDIO_PATH)
    phase_coherence = calculate_phase_coherence(audio)
    assert 0 <= phase_coherence <= 1, "Phase coherence should be between 0 and 1"

def test_train_model():
    audio_files = [TEST_AUDIO_PATH]
    model_save_path = 'tests/model_epoch_{}.pth'
    os.makedirs('tests', exist_ok=True)

    # Minimal run to test function call, not actual training
    train_model(
        audio_files,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        epochs=1,  # Only 1 epoch for testing
        batch_size=BATCH_SIZE,
        model_save_path=model_save_path
    )

    # Check if model file is created
    assert os.path.exists(model_save_path.format(1)), "Model file was not saved"

if __name__ == "__main__":
    pytest.main([__file__])
