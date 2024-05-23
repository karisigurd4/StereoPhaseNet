import sys
import os

# Add the root directory of the project to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.train import train_model
from src.config import *
import glob

def main():
    # Example audio file paths
    audio_files = glob.glob(os.path.join(project_root, 'data', '*.mp3'))
    
    train_model(
        audio_files, 
        frame_length=FRAME_LENGTH, 
        hop_length=HOP_LENGTH, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )

if __name__ == "__main__":
    main()
