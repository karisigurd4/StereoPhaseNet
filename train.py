import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import AudioCNN
from data_processing import load_and_normalize_audio, generate_out_of_phase, frame_entire_audio
import csv
import librosa

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
        
def calculate_phase_coherence(audio):
    left_channel, right_channel = audio
    stft_left = librosa.stft(left_channel)
    stft_right = librosa.stft(right_channel)
    phase_left = np.angle(stft_left)
    phase_right = np.angle(stft_right)
    phase_diff = phase_left - phase_right
    phase_coherence = np.abs(np.mean(np.cos(phase_diff)))
    return phase_coherence

def train_model(audio_files, frame_length=256, hop_length=128, epochs=10, batch_size=8, model_save_path="model/model_epoch_{}.pth"):
    dataset = AudioDataset(audio_files, frame_length=frame_length, hop_length=hop_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioCNN(input_length=frame_length).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # Initialize CSV logging
    log_file = "training_log.csv"
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Phase Coherence"])

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        phase_coherence_sum = 0.0
        num_batches = 0

        for batch_idx, (frames, targets) in enumerate(dataloader):
            frames = frames.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate phase coherence for logging
            outputs_np = outputs.cpu().detach().numpy()
            targets_np = targets.cpu().detach().numpy()
            batch_phase_coherence = calculate_phase_coherence((outputs_np[:, 0, :], outputs_np[:, 1, :]))
            phase_coherence_sum += batch_phase_coherence
            num_batches += 1

        avg_loss = epoch_loss / len(dataloader)
        avg_phase_coherence = phase_coherence_sum / num_batches
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}, Phase Coherence: {avg_phase_coherence}')

        # Log to CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss, avg_phase_coherence])

        # Step the scheduler
        scheduler.step(avg_loss)

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), model_save_path.format(epoch + 1))
