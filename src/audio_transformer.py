import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2ForPhaseCorrection(nn.Module):
    def __init__(self, input_length=16000, hidden_size=512):
        super(Wav2Vec2ForPhaseCorrection, self).__init__()
        self.input_length = input_length
        self.hidden_size = hidden_size
        
        # Load Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Define the fully connected layers for output
        self.fc1 = nn.Linear(self.wav2vec2.config.hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * input_length // 160)  # Adjust to match the input length

    def forward(self, x):
        # Expect x to be [batch_size, num_channels, seq_length]
        batch_size, num_channels, seq_length = x.shape
        
        # Permute and reshape to match Wav2Vec2 expected input
        x = x.permute(0, 2, 1).reshape(batch_size, seq_length * num_channels)  # Shape: [batch_size, seq_length * num_channels]

        # Apply Wav2Vec2 model
        outputs = self.wav2vec2(x).last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]

        # Apply fully connected layers
        x = torch.relu(self.fc1(outputs))  # Shape: [batch_size, seq_length, hidden_size]
        x = self.fc2(x)  # Shape: [batch_size, seq_length, 2 * input_length // 160]

        # Reshape to [batch_size, num_channels, seq_length]
        x = x.permute(0, 2, 1).reshape(batch_size, 2, -1)
        
        return x