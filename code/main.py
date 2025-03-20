import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def create_spectrogram(wav_file, output_folder):
    rate, data = wav.read(wav_file)
    plt.specgram(data, Fs=rate)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    output_file = os.path.join(output_folder, os.path.basename(wav_file).replace('.wav', '.png'))
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            wav_file = os.path.join(input_folder, file_name)
            create_spectrogram(wav_file, output_folder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process WAV files to create spectrograms.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing WAV files')
    parser.add_argument('output_folder', type=str, help='Path to the folder to save spectrograms')

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)