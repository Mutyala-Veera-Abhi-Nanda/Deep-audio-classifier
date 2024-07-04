# Deep Audio Classifier Project

This project is for classifying audio files using deep learning techniques. The project involves preprocessing audio data, training a model, and evaluating its performance.

## Requirements:

- Python 3.x

- numpy

- pandas

- tensorflow

- librosa

- matplotlib

## Project Overview:

The goal of this project is to develop a model that can accurately classify audio files into predefined categories. This involves several steps:

#### Algorithm Description

- Dataset: The dataset includes various audio clips labeled as Capuchin bird calls, non-Capuchin bird calls, and other forest sounds. The data come from a well-defined source Kaggle competition dataset.

- Steps:

1. Convert audio data to waveforms: Audio data is read and converted into a format (waveform) that can be processed.

2. Transform waveforms into spectrograms: Spectrograms provide a visual representation of the audio signal's frequency spectrum over time.

3. Classify Capuchin bird calls using the transformed data: A CNN model processes the spectrograms to classify the audio clips.

#### Data Preparation

- Dependencies: Installation of libraries such as TensorFlow, TensorFlow I/O (for audio processing), and Matplotlib for plotting.
- Downloading Dataset: The dataset is downloaded from Kaggle using the Kaggle API.
- File Handling: Instructions for copying Kaggle API credentials and setting directory permissions to access the dataset.

#### Dataset Extraction and Initial Setup

- Extraction: The downloaded dataset, usually in a compressed format like zip, is extracted to a specified directory.
- File Paths: Paths to audio files are set up for easy access during preprocessing and model training. These paths include directories for Capuchin calls, non-Capuchin calls datasets.

#### Preprocessing

Downsampling: The audio clips, originally recorded at a high sample rate (44,100 Hz), are downsampled to 16,000 Hz. This reduces the computational load and ensures consistency.
Reason for Downsampling: Downsampling helps in reducing the size of the audio data, making it easier and faster to process without significant loss of important audio features.

#### Audio Loading Function

- Function Definition: load_wav_16k_mono(filename) is designed to load an audio file and resample it to 16,000 Hz in mono channel format.
- Resampling: Adjusts the sample rate of the audio.
- Mono Conversion: Converts stereo audio to mono, which simplifies the data without affecting the detection of bird calls.

#### Model Training (Implied Steps)

- Spectrogram Conversion: Transforming the waveform data into spectrograms is crucial as CNNs excel at processing 2D image-like data. Spectrograms represent the frequency (y-axis) and time (x-axis) with amplitude shown by varying colors.
- CNN Architecture: Typically involves:
- Convolutional Layers: To extract features from the spectrograms.
- Pooling Layers: To reduce dimensionality and highlight important features.
- Fully Connected Layers: For classification based on the extracted features.
