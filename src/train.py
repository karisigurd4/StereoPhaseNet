import torch
from torch.utils.data import DataLoader
from src.audio_transformer import AudioTransformer
from sklearn.model_selection import train_test_split
from src.audio_dataset import AudioDataset

def train_model(input_files, target_files, frame_length=512, hop_length=256, epochs=10, batch_size=8, model_save_path="../model/model_epoch_{}.pth"):
    # Check if input_files and target_files have the same length
    if len(input_files) != len(target_files):
        raise ValueError("The number of input files and target files must be the same.")

    train_input_files, val_input_files, train_target_files, val_target_files = train_test_split(input_files, target_files, test_size=0.2, random_state=42)
    
    train_dataset = AudioDataset(train_input_files, train_target_files, frame_length=frame_length, hop_length=hop_length)
    val_dataset = AudioDataset(val_input_files, val_target_files, frame_length=frame_length, hop_length=hop_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = AudioTransformer(input_length=frame_length).cuda() 
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (frames, targets) in enumerate(train_dataloader):
            frames = torch.tensor(frames, dtype=torch.float32).cuda()  
            targets = torch.tensor(targets, dtype=torch.float32).cuda()  

            optimizer.zero_grad()
            outputs = model(frames)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, targets in val_dataloader:
                frames = torch.tensor(frames, dtype=torch.float32).cuda() 
                targets = torch.tensor(targets, dtype=torch.float32).cuda() 
                
                outputs = model(frames)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Learning Rate: {current_lr}')

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