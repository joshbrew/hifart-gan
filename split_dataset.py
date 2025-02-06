import os
import random

# Define dataset paths
mel_dir = "dataset/mels"
wav_dir = "dataset/wavs"

train_file = "dataset/train_files.txt"
val_file = "dataset/validation_files.txt"

# Set train/validation split ratio
VAL_SPLIT = 0.1  # 10% validation

# Get sorted list of files
wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
mel_files = sorted([f for f in os.listdir(mel_dir) if f.endswith(".npy")])

# Ensure matching WAV and Mel files
if len(wav_files) != len(mel_files):
    print("❌ Error: Number of WAVs and Mel spectrograms do not match!")
    exit()

# Pair them together
file_pairs = list(zip(wav_files, mel_files))

# Shuffle dataset for randomness
random.shuffle(file_pairs)

# Split dataset
val_size = int(len(file_pairs) * VAL_SPLIT)
val_pairs = file_pairs[:val_size]
train_pairs = file_pairs[val_size:]

# Write training file list
with open(train_file, "w") as f_train:
    for wav_file, mel_file in train_pairs:
        wav_path = os.path.join(wav_dir, wav_file)
        mel_path = os.path.join(mel_dir, mel_file)
        f_train.write(f"{wav_path}|{mel_path}\n")

# Write validation file list
with open(val_file, "w") as f_val:
    for wav_file, mel_file in val_pairs:
        wav_path = os.path.join(wav_dir, wav_file)
        mel_path = os.path.join(mel_dir, mel_file)
        f_val.write(f"{wav_path}|{mel_path}\n")

print(f"✅ Split completed! {len(train_pairs)} training samples, {len(val_pairs)} validation samples.")

