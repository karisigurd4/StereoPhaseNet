import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class AudioTransformer(nn.Module):
    def __init__(self, input_length=512, hidden_size=512):
        super(AudioTransformer, self).__init__()
        self.input_length = input_length
        self.hidden_size = hidden_size
        
        # Define the embedding layer to project from num_channels (2) to hidden_size
        self.embedding = nn.Linear(2, hidden_size)
        
        # Define the BERT model
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=input_length,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.encoder = BertModel(config)
        
        # Fully connected layers for output
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)  # Predicting 2 channels

    def forward(self, x):
        # Expect x to be [batch_size, num_channels, seq_length]
        batch_size, num_channels, seq_length = x.shape
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, seq_length, num_channels]

        # Apply embedding layer to project to hidden_size
        x = self.embedding(x)  # Shape: [batch_size, seq_length, hidden_size]

        # Apply transformer encoder
        encoder_output = self.encoder(inputs_embeds=x).last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]

        # Apply fully connected layers while maintaining the sequence structure
        x = torch.relu(self.fc1(encoder_output))  # Shape: [batch_size, seq_length, hidden_size]
        x = self.fc2(x)  # Shape: [batch_size, seq_length, 2]

        # Reshape to [batch_size, num_channels, seq_length]
        x = x.permute(0, 2, 1)  # Shape: [batch_size, 2, seq_length]

        return x