import os

mel_dir = "dataset/mels"
wav_dir = "dataset/wavs"
train_file = "dataset/train_files.txt"

with open(train_file, "w") as f:
    for mel_file in os.listdir(mel_dir):
        if mel_file.endswith(".npy"):
            wav_file = mel_file.replace(".npy", ".wav")
            mel_path = os.path.join(mel_dir, mel_file)
            wav_path = os.path.join(wav_dir, wav_file)
            if os.path.exists(wav_path):
                f.write(f"{wav_path}|{mel_path}\n")
            else:
                print(f"⚠ Missing WAV file: {wav_path}")

print(f"✅ Training file list recreated: {train_file}")

