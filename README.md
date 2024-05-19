# StereoSync: Phase Correction for Stereo Audio Using Deep Learning

## Introduction

StereoSync is a deep learning project aimed at correcting phase issues in stereo audio recordings. The project focuses on creating a dataset of out-of-phase stereo audio stems and their in-phase counterparts, and then training a neural network to convert out-of-phase audio to in-phase audio. This ensures a balanced and full sound when played in both stereo and mono setups, enhancing the overall audio quality.

## Project Description

### Motivation

In audio production, phase issues between the left and right channels of stereo recordings can lead to undesirable effects such as frequency cancellations and a hollow sound. These problems become especially prominent when the audio is played back in mono. StereoSync aims to solve this problem by leveraging deep learning techniques to automatically correct phase discrepancies.

### Objectives

- **Data Collection**: Gather a diverse set of high-quality stereo audio files to create a comprehensive dataset.
- **Data Preparation**: Process the audio files to generate pairs of out-of-phase and in-phase audio samples.
- **Model Development**: Design and train a convolutional neural network (CNN) to learn the mapping from out-of-phase to in-phase audio.
- **Evaluation**: Assess the model's performance using appropriate metrics and fine-tune it for optimal results.
- **Deployment**: Develop an inference pipeline that can be used to correct phase issues in new audio recordings.

### Features

- **Audio Processing**: Efficient loading, normalization, and framing of stereo audio files.
- **Deep Learning**: CNN-based model tailored for audio phase correction.
- **Training and Evaluation**: Comprehensive training loop with data loaders and evaluation metrics.
- **Inference Pipeline**: Ready-to-use pipeline for applying the trained model to new audio data.

### Getting Started

Follow the [setup instructions](#stereosync-project-setup-instructions) to create the environment and install all necessary dependencies. Then, use the provided scripts to preprocess your audio data, train the model, and evaluate its performance.

### Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality of this project.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

By correcting phase issues in stereo audio recordings, StereoSync aims to improve the listening experience and ensure high-quality sound reproduction in any playback setup.

## StereoSync Project Setup Instructions

### Step 1: Create a New Anaconda Environment

1. Open your terminal (or Anaconda Prompt on Windows).
2. Create a new environment named `stereosync` with Python 3.8:

    ```sh
    conda create --name stereosync python=3.8
    ```

3. Activate the new environment:

    ```sh
    conda activate stereosync
    ```

### Step 2: Install Required Packages

1. **Librosa**: For audio processing.
2. **PyTorch**: Deep learning framework.
3. **Other dependencies**: NumPy, etc.

Install the packages using the following commands:

    ```sh
    conda install -c conda-forge librosa 
    conda install ffmpeg pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia 
    ```

### Step 3: Verify Installation

1. Verify the installation of each package:

    ```sh
    python -c "import librosa; print('librosa installed')"
    python -c "import torch; print('PyTorch installed')"
    ```

### Step 4: Optional - Jupyter Notebook

If you plan to use Jupyter Notebook for experiments and development:

1. Install Jupyter:

    ```sh
    conda install -c conda-forge notebook
    ```

2. Launch Jupyter Notebook:

    ```sh
    jupyter notebook
    ```

### Summary of Commands

Here’s a summary of all the commands to set up your environment and install the necessary packages:

    ```sh
    # Create and activate the new environment
    conda create --name stereosync python=3.8
    conda activate stereosync

    # Install required packages
    conda install -c conda-forge librosa
    conda install ffmpeg pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia

    # Optional: Install Jupyter Notebook
    conda install -c conda-forge notebook

    # Verify installations
    python -c "import librosa; print('librosa installed')"
    python -c "import torch; print('PyTorch installed')"
    ```

By following these steps, you’ll have a fully set up environment ready for your StereoSync project.
