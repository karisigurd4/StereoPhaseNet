import sys
import os

# Add the root directory of the project to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.train import train_model
import glob

def main():
    input_audio_files = glob.glob(os.path.join(project_root, 'data/unprocessed/', '*.wav'))
    processed_audio_files = glob.glob(os.path.join(project_root, 'data/processed/', '*.wav'))

    train_model(
        input_audio_files, 
        processed_audio_files,
        frame_length=2048,
        hop_length=1024, 
        epochs=10, 
        batch_size=8,
        model_save_path="../model/model_epoch_{}.pth"
    )

if __name__ == "__main__":
    main()
