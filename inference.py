import torch
import numpy as np
import librosa
import soundfile as sf
from audio_cnn import AudioCNN
from data_processing import load_and_normalize_audio, generate_out_of_phase, frame_entire_audio
from audio_transformer import AudioTransformer

# Load the trained model
def load_model(model_path):
    model = AudioTransformer(input_length=512).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Preprocess the input audio file
def preprocess_audio(file_path):
    audio, sr = load_and_normalize_audio(file_path)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("Input audio must be stereo with shape (2, samples)")

    out_of_phase_audio = generate_out_of_phase(audio)
    return out_of_phase_audio, sr, audio

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

# Run inference on the audio file
def run_inference(model, file_path, output_path):
    out_of_phase_audio, sr, original_audio = preprocess_audio(file_path)
    out_of_phase_frames = frame_entire_audio(out_of_phase_audio, frame_length=512, hop_length=256)
    
    in_phase_predictions = []

    with torch.no_grad():
        for i in range(0, len(out_of_phase_frames), 8):
            batch = out_of_phase_frames[i:i + 8]
            batch = torch.tensor(batch, dtype=torch.float32).cuda()
            print(f"Batch shape before model: {batch.shape}")  # Debug print to check batch shape
            output = model(batch)
            print(f"Output shape from model: {output.shape}")  # Debug print to check output shape
            in_phase_predictions.extend(output.cpu().numpy())

    in_phase_predictions = np.array(in_phase_predictions)
    print(f"Final in_phase_predictions shape: {in_phase_predictions.shape}")  # Debug print to check final predictions shape

    # Ensure the shape is correct for librosa.istft
    in_phase_predictions = in_phase_predictions.transpose(1, 0, 2)  # Shape: (2, num_frames, frame_length)
    print(f"in_phase_predictions transposed shape: {in_phase_predictions.shape}")

    # Combine the frames back into a single audio signal manually
    num_frames = in_phase_predictions.shape[1]
    frame_length = in_phase_predictions.shape[2]
    hop_length = 256
    output_length = hop_length * (num_frames - 1) + frame_length

    left_channel = np.zeros(output_length)
    right_channel = np.zeros(output_length)
    window = np.hanning(frame_length)

    for i in range(num_frames):
        start = i * hop_length
        left_channel[start:start + frame_length] += in_phase_predictions[0, i, :] * window
        right_channel[start:start + frame_length] += in_phase_predictions[1, i, :] * window

    # Normalize the audio to prevent clipping and numerical instability
    # max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    # if max_val > 1.0:
    #     left_channel /= max_val
    #     right_channel /= max_val

    in_phase_audio = np.vstack((left_channel, right_channel))

    print(f"Length of original audio: {len(original_audio[0])}")  # Debug print for original audio length
    print(f"Length of in-phase audio: {len(in_phase_audio[0])}")  # Debug print for in-phase audio length

    # Save the output audio file using soundfile
    sf.write(output_path, in_phase_audio.T, sr)  # Note the transpose to match (samples, channels) format

    # Calculate and print phase coherence
    original_phase_coherence = calculate_phase_coherence(original_audio)
    output_phase_coherence = calculate_phase_coherence(in_phase_audio)

    print(f"Original Phase Coherence: {original_phase_coherence:.4f}")
    print(f"Output Phase Coherence: {output_phase_coherence:.4f}")

if __name__ == "__main__":
    model_path = "model/model_epoch_4.pth"
    input_file = "input_audio_file.mp3"
    output_file = "output_audio_file.mp3"

    model = load_model(model_path)
    run_inference(model, input_file, output_file)
