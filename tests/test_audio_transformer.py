import pytest
import torch
from src.models.audio_transformer import AudioTransformer

# Define test parameters
BATCH_SIZE = 8
NUM_CHANNELS = 2
SEQ_LENGTH = 512
HIDDEN_SIZE = 512

def test_audio_transformer_forward():
    # Instantiate the model
    model = AudioTransformer(input_length=SEQ_LENGTH, hidden_size=HIDDEN_SIZE)

    # Create a dummy input tensor [batch_size, num_channels, seq_length]
    dummy_input = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)

    # Run the forward pass
    output = model(dummy_input)

    # Check the output shape
    assert output.shape == (BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH), f"Output shape is incorrect. Expected ({BATCH_SIZE}, {NUM_CHANNELS}, {SEQ_LENGTH}), got {output.shape}"

    # Check the model's device
    assert dummy_input.device == output.device, "Input and output tensors are not on the same device"

if __name__ == "__main__":
    pytest.main([__file__])
