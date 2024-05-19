import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import AudioCNN
from data_processing import load_and_normalize_audio, generate_out_of_phase, frame_entire_audio

class AudioDataset(Dataset):
    def __init__(self, audio_files, frame_length=512, hop_length=256):
        self.frames = []
        self.targets = []

        for file_path in audio_files:
            audio, _ = load_and_normalize_audio(file_path)
            out_of_phase_audio = generate_out_of_phase(audio)
            
            frames = frame_entire_audio(out_of_phase_audio, frame_length=frame_length, hop_length=hop_length)
            targets = frame_entire_audio(audio, frame_length=frame_length, hop_length=hop_length)
            
            self.frames.extend(frames)
            self.targets.extend(targets)

        self.frames = np.array(self.frames)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.frames[idx], self.targets[idx])
        
def train_model(audio_files, frame_length=64, hop_length=32, epochs=10, batch_size=8, model_save_path="stereosync_model.pth"):
    dataset = AudioDataset(audio_files, frame_length=frame_length, hop_length=hop_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioCNN(input_length=frame_length).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (frames, targets) in enumerate(dataloader):
            frames = frames.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')

    torch.save(model.state_dict(), model_save_path)