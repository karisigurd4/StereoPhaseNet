from train import train_model
from config import *
import glob

def main():
    # Example audio file paths
    audio_files = glob.glob('data/*.mp3')
    
    train_model (
        audio_files, 
        frame_length=FRAME_LENGTH, 
        hop_length=HOP_LENGTH, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )

if __name__ == "__main__":
    main()
