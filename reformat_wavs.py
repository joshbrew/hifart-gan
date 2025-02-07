import os
from pydub import AudioSegment

wav_dir = "./dataset/wavs"

for file in os.listdir(wav_dir):
    if file.endswith(".wav"):
        file_path = os.path.join(wav_dir, file)
        
        # Load WAV file
        audio = AudioSegment.from_file(file_path)
        
        # Convert to 16-bit PCM WAV
        audio = audio.set_frame_rate(22050).set_sample_width(2).set_channels(1)
        
        # Save the new file
        audio.export(file_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        
        print(f"✅ Converted: {file}")

print("🎯 All WAVs are now 16-bit PCM.")

