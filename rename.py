import os

# Define dataset paths
wav_dir = "dataset/wavs"
mel_dir = "dataset/mels"

# Ensure the directories exist
if not os.path.exists(wav_dir) or not os.path.exists(mel_dir):
    print("❌ Error: One or both dataset folders are missing!")
    exit()

# Get and sort the list of files
wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
mel_files = sorted([f for f in os.listdir(mel_dir) if f.endswith(".npy")])

# Ensure the number of WAVs matches the number of Mels
if len(wav_files) != len(mel_files):
    print("❌ Error: The number of WAV files and Mel spectrograms does not match!")
    exit()

# Rename files sequentially
for i, (wav_file, mel_file) in enumerate(zip(wav_files, mel_files), start=1):
    new_wav_name = f"{i:04d}.wav"  # e.g., 0001.wav, 0002.wav
    new_mel_name = f"{i:04d}.npy"  # e.g., 0001.npy, 0002.npy

    # Rename WAV file
    os.rename(os.path.join(wav_dir, wav_file), os.path.join(wav_dir, new_wav_name))

    # Rename Mel spectrogram file
    os.rename(os.path.join(mel_dir, mel_file), os.path.join(mel_dir, new_mel_name))

print(f"✅ Renamed {len(wav_files)} files sequentially!")

