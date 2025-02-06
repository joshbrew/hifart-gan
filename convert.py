import os
import numpy as np
import librosa
import librosa.display

# Define input and output directories
INPUT_DIR = os.path.expanduser("~/Desktop/audio/")  # Path to your WAV files
OUTPUT_DIR = os.path.expanduser("~/Desktop/mel_spectrograms/")  # Path to save mel spectrograms

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_wav_to_mel(wav_path, output_path, sr=22050, n_mels=80):
    """Convert a WAV file to a Mel spectrogram and save it as a NumPy file."""
    try:
        # Load WAV file
        y, sr = librosa.load(wav_path, sr=sr)

        # Convert to Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

        # Convert to dB scale for better visualization (optional)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Save as NumPy array
        np.save(output_path, mel_spectrogram)
        print(f"✅ Saved Mel Spectrogram: {output_path}")
    except Exception as e:
        print(f"❌ Error processing {wav_path}: {e}")

def scan_and_process_audio(input_dir, output_dir):
    """Scan the directory for WAV files and convert them to mel spectrograms."""
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    
    if not wav_files:
        print("⚠ No WAV files found in the input directory.")
        return
    
    print(f"🔍 Found {len(wav_files)} WAV files. Converting to Mel spectrograms...")

    for wav_file in wav_files:
        wav_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, wav_file.replace(".wav", ".npy"))
        convert_wav_to_mel(wav_path, output_path)

# Run the script
scan_and_process_audio(INPUT_DIR, OUTPUT_DIR)

