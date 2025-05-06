#File to audio signal
from matplotlib import pyplot as plta
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np
import pandas as pd
import os
import pickle

import librosa
import soundfile as sf
import sounddevice as sd
import noisereduce as nr

    # Constants
# Global variables
SAMPLE_RATE = 44100 # Value in most audio files
MAX_LENGTH = SAMPLE_RATE * 10 # Assuming audios are 10 seconds long
FRAME_SIZE = 2048
HOP_LENGTH = 256 # Lower val = higher res
N_MELS = 256
MIN_VAL = 0  # To normalize
MAX_VAL = 1

def load_audio(file_path):
    signal = librosa.load(file_path, sr=SAMPLE_RATE)[0]
    return signal

def apply_padding(array):
    if len(array) < MAX_LENGTH:
        num_missing_items = MAX_LENGTH - len(array)
        padded_array = np.pad(array, (num_missing_items // 2, num_missing_items // 2), mode='constant')
        return padded_array
    elif len(array) > MAX_LENGTH:
        center = len(array) // 2
        start = max(0, center - MAX_LENGTH // 2)
        end = min(len(array), start + MAX_LENGTH)
        trimmed_array = array[start:end]
        return trimmed_array
    return array

def min_max_normalize(array):
    max_val = array.max()
    min_val = array.min()
    norm_array = (array - min_val) / (max_val - min_val)
    return norm_array, min_val, max_val

def denormalize(norm_array, original_min, original_max):
    #array = (norm_array - min_val) / (max_val - min_val)
    #array = array * (original_max - original_min) + original_min
    array = norm_array * (original_max - original_min) + original_min
    return array
#STFT instead
def compute_stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH):
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    magnitude = min_max_normalize(magnitude)
    phase = np.angle(stft)
    return magnitude, phase

def audio_from_stft(magnitude, phase, hop_length=HOP_LENGTH):
    phase = np.unwrap(phase, axis=-1)
    complex_stft = magnitude * np.exp(1j * phase)
    signal = librosa.istft(complex_stft, hop_length=hop_length)
    return signal

# Plays audio
# Find out output_device_id value for ur device using print(sd.query_devices())
def play_audio(audio, output_device_id=4):
    Audio(audio, rate=44100)
    #sd.play(audio, samplerate=SAMPLE_RATE, device=output_device_id)
    #sd.wait()

if __name__ == "__main__":

    # Load the audio file
    file_path = "D:/Development/audioGen/SythesizedBreathingM-3/data/Test.wav"
    signal = load_audio(file_path)

    # Apply padding to the signal
    padded_signal = apply_padding(signal)

    # Compute STFT
    magnitude, phase = compute_stft(padded_signal)

    # Reconstruct the audio from STFT
    reconstructed_signal = audio_from_stft(magnitude, phase)

    # Save the reconstructed audio to a file
    output_file_path = "./output/reconstructed.wav"
    sf.write(output_file_path, reconstructed_signal, SAMPLE_RATE)