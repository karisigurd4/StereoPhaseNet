from train import train_model
from config import *
import glob

def main():
    # Example audio file paths
    audio_files = glob.glob('data/*.mp3')
    
    # Path to save the trained model
    model_save_path = 'model/stereosync_model.pth'
    
    train_model(audio_files, model_save_path=model_save_path)

if __name__ == "__main__":
    main()
