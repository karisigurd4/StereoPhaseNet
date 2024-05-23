import torch
from torch.utils.data import DataLoader
from src.audio_transformer import Wav2Vec2ForPhaseCorrection
import csv
from src.data_processing import load_and_normalize_audio, generate_out_of_phase, frame_entire_audio
import numpy as np
import librosa
import datetime
from sklearn.model_selection import train_test_split

class AudioDataset:
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

# Calculate phase coherence
def calculate_phase_coherence(audio):
    left_channel, right_channel = audio
    stft_left = librosa.stft(left_channel)
    stft_right = librosa.stft(right_channel)
    phase_left = np.angle(stft_left)
    phase_right = np.angle(stft_right)
    phase_diff = phase_left - phase_right
    phase_coherence = np.abs(np.mean(np.cos(phase_diff)))
    return phase_coherence

def train_model(audio_files, frame_length=16000, hop_length=8000, epochs=10, batch_size=8, model_save_path="../model/model_epoch_{}.pth"):
    train_files, val_files = train_test_split(audio_files, test_size=0.2, random_state=42)
    
    train_dataset = AudioDataset(train_files, frame_length=frame_length, hop_length=hop_length)
    val_dataset = AudioDataset(val_files, frame_length=frame_length, hop_length=hop_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Wav2Vec2ForPhaseCorrection(input_length=frame_length).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Initialize CSV logging
    log_file = "../logs/training_log.csv"
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Phase Coherence", "Learning Rate"])

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        phase_coherence_sum = 0.0
        num_batches = 0

        for batch_idx, (frames, targets) in enumerate(train_dataloader):
            frames = frames.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(frames)
            
            # Resize targets to match the outputs
            batch_size, num_channels, seq_length = outputs.shape

            # Ensure the targets and outputs have compatible shapes
            targets = targets[:, :, :seq_length]

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

        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_phase_coherence = phase_coherence_sum / num_batches

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, targets in val_dataloader:
                frames = frames.cuda()
                targets = targets.cuda()
                outputs = model(frames)
                
                # Ensure targets match the sequence length of outputs
                if outputs.shape[2] != targets.shape[2]:
                    targets = targets[:, :, :outputs.shape[2]]
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Phase Coherence: {avg_phase_coherence}, Learning Rate: {current_lr}')

        # Log to CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_phase_coherence, current_lr])

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path.format(epoch + 1))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), model_save_path.format(epoch + 1))
