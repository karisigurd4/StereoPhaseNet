import sys
import os

# Add the root directory of the project to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import wave
import torch
import numpy as np
import librosa
import soundfile as sf
from src.data_processing import load_and_normalize_audio, frame_entire_audio, apply_fft
from src.audiofft_cnn import AudioFFT_CNN
import matplotlib.pyplot as plt

inference_frame_length = 1024
inference_hop_length = 512

# Load the trained model
def load_model(model_path, frame_length):
    model = AudioFFT_CNN(input_length=frame_length).cuda()  
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Preprocess the input audio file
def preprocess_audio(file_path):
    audio, sr = load_and_normalize_audio(file_path)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("Input audio must be stereo with shape (2, samples)")

    return audio, sr

def output_audio(frames, output_path, sr):
    # Combine the frames back into a single audio signal using overlap-add method
    num_frames, num_channels, frame_length = frames.shape
    signal_length = inference_hop_length * (num_frames - 1) + frame_length

    combined_signal = np.zeros((num_channels, signal_length))
    overlap_count = np.zeros(signal_length)

    for i in range(num_frames):
        start = i * inference_hop_length
        end = start + frame_length
        combined_signal[:, start:end] += frames[i, :, :]
        overlap_count[start:end] += 1

    # Avoid dividing by zero in non-overlapping regions
    overlap_count[overlap_count == 0] = 0.00001
    combined_signal /= overlap_count

    # Save the output audio file using soundfile
    sf.write(output_path, combined_signal.T, sr)  # Note the transpose to match (samples, channels) format

# Run inference on the audio file
def run_inference(model, file_path, output_path):
    original_audio, sr = preprocess_audio(file_path)
    
    # Frame the entire audio
    input_frames = frame_entire_audio(original_audio, frame_length=inference_frame_length, hop_length=inference_hop_length)
    
    # Initialize the lists to store model outputs
    delta_prediction_frames = []

    with torch.no_grad():
        for i in range(0, len(input_frames), 8):
            batch = input_frames[i:i + 8]
            batch = torch.tensor(batch, dtype=torch.float32).cuda()

            output = model(batch)

            delta_prediction_frames.extend(output.cpu().numpy())

    delta_prediction_frames = np.array(delta_prediction_frames)

    # Select the data for delta_fft_predictions[0, 0, :]
    data_to_plot = delta_prediction_frames[0, 0, :]

    # Plotting the selected data as a line chart
    plt.figure(figsize=(12, 6))
    plt.plot(data_to_plot)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Delta FFT Value')
    plt.title('Delta FFT Predictions for Frame 0, Channel 0')
    plt.show()

    # Convert FFT frames back to time domain using inverse FFT
    processed_frames = input_frames + delta_prediction_frames

    output_audio(processed_frames, output_file, sr)
    output_audio(delta_prediction_frames, "../delta.wav", sr)

if __name__ == "__main__":
    model_path = "../model/model_epoch_1.pth"
    input_file = "../input_audio_file.wav"
    output_file = "../output_audio_file.wav"

    model = load_model(model_path, inference_frame_length)
    run_inference(model, input_file, output_file)
