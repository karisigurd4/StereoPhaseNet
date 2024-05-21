from train import train_model
from config import *
import glob

def main():
    # Example audio file paths
    audio_files = glob.glob('data/*.mp3')
    
    train_model(audio_files)

if __name__ == "__main__":
    main()
