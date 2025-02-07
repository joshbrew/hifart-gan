import numpy as np
import torchaudio

# Load the original audio file
wav_file_path = "./dataset/wavs/0058.wav"
waveform, sr = torchaudio.load(wav_file_path)
original_duration = waveform.shape[1] / sr  # Duration in seconds
wav_file_path2 = "./generated_audio.wav"
waveform2, sr = torchaudio.load(wav_file_path2)
generated_duration = waveform2.shape[1] / sr  # Duration in seconds

# Load the spectrogram file
spectrogram_file_path = "./dataset/mels/0058.npy"
spectrogram = np.load(spectrogram_file_path)
num_mels, num_time_steps = spectrogram.shape

# Test common hop sizes
hop_sizes = [256, 512, 128, 275]  # Common values for vocoders
duration_estimates = {hop: (num_time_steps * hop) / sr for hop in hop_sizes}

print(f"Original Audio Duration: {original_duration:.2f} sec")
print(f"Spectrogram Time Steps: {num_time_steps}")
print(f"Estimated Durations for different hop sizes: {duration_estimates}")
print(f"Generated Audio Duration: {generated_duration:.2f} sec")
