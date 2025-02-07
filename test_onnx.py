import torch
import torchaudio
import argparse
import onnxruntime as ort
import numpy as np
from scipy.signal import lfilter

# If using spectral denoising, import noisereduce (install via pip install noisereduce)
try:
    import noisereduce as nr
except ImportError:
    nr = None

def load_mel_spectrogram(file_path,
                         apply_log_scale=False,
                         normalize_mel=False):
    """
    Load a mel spectrogram from a .npy file and apply preprocessing.

    Parameters:
      file_path (str): Path to the .npy mel spectrogram.
      apply_log_scale (bool): If True, apply logarithmic scaling.
      normalize_mel (bool): If True, normalize using preset mean and std.

    Adjust the parameters below to match your training configuration.
    """
    mel = np.load(file_path).astype(np.float32)
    print(f"Loaded Spectrogram Shape: {mel.shape}")
    num_mels, num_time_steps = mel.shape
    if num_mels != 80:
        raise ValueError(f"Expected mel spectrogram with 80 mel bins, but got {num_mels}")

    # Optionally apply logarithmic scaling.
    if apply_log_scale:
        eps = 1e-5
        mel = np.log10(np.maximum(mel, eps))
        print("Applied logarithmic scaling to the mel spectrogram.")
    else:
        print("Logarithmic scaling not applied.")

    # Optionally normalize the mel spectrogram.
    if normalize_mel:
        # Replace these values with those used during training.
        mel_mean = -4.0  
        mel_std = 4.0    
        mel = (mel - mel_mean) / (mel_std + 1e-9)
        print(f"Applied normalization (mean={mel_mean}, std={mel_std}).")
    else:
        print("Normalization not applied.")

    return mel

def apply_deemphasis(signal, coeff=0.95):
    """
    Apply a de–emphasis filter to the signal.

    This reverses a simple pre–emphasis that may have been applied during training.
    The filter is defined as: y[n] = x[n] + coeff * y[n-1]
    """
    return lfilter([1], [1, -coeff], signal)

def normalize_audio(signal, target_peak=0.99):
    """
    Normalize the audio signal so that its maximum absolute amplitude equals target_peak.
    """
    peak = np.max(np.abs(signal))
    if peak > 0:
        return signal * (target_peak / peak)
    return signal

def generate_audio(onnx_path,
                   output_path,
                   spectrogram_path,
                   sampling_rate=22050,
                   hop_size=256,
                   device='cpu',
                   apply_log_scale=False,
                   normalize_mel=False,
                   output_gain=1.0,
                   apply_deemph=False,
                   normalize_final=False,
                   apply_denoiser=False,
                   denoiser_strength=0.1,
                   apply_spectral_denoise=False,
                   spectral_denoise_prop_decrease=1.0,
                   final_boost=1.0):
    """
    Generate an audio clip using the exported ONNX model.

    Parameters:
      - onnx_path (str): Path to the ONNX generator model.
      - output_path (str): Where the generated audio will be saved.
      - spectrogram_path (str): Path to the mel spectrogram (.npy file).
      - sampling_rate (int): Sampling rate for output audio.
      - hop_size (int): Hop size used during spectrogram generation.
      - device (str): 'cpu' or 'cuda' for inference.
      - apply_log_scale (bool): Whether to apply logarithmic scaling.
      - normalize_mel (bool): Whether to normalize the mel spectrogram.
      - output_gain (float): Factor to scale the output audio amplitude.
      - apply_deemph (bool): Whether to apply a de–emphasis filter.
      - normalize_final (bool): Whether to normalize the final audio amplitude.
      - apply_denoiser (bool): Whether to subtract a vocoder bias to denoise.
      - denoiser_strength (float): Strength multiplier for bias subtraction.
      - apply_spectral_denoise (bool): Whether to apply spectral denoising (requires noisereduce).
      - spectral_denoise_prop_decrease (float): Controls the amount of noise reduction.
      - final_boost (float): Additional multiplier applied at the very end.
    """
    # Initialize ONNX runtime.
    providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    # Load and preprocess the mel spectrogram.
    mel = load_mel_spectrogram(spectrogram_path,
                               apply_log_scale=apply_log_scale,
                               normalize_mel=normalize_mel)
    num_mels, num_time_steps = mel.shape

    # Reshape to match model expectations: [batch, num_mels, time_steps]
    mel_input = mel.reshape(1, num_mels, num_time_steps)
    print(f"Final Input Spectrogram Shape: {mel_input.shape}")

    # Run inference.
    audio_output = session.run(None, {'mel': mel_input})[0]
    audio_tensor = torch.tensor(audio_output)

    # Remove extra dimensions to get shape [channels, time_samples].
    if audio_tensor.ndim == 3:
        audio_tensor = audio_tensor.squeeze(0).squeeze(0)
    elif audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.squeeze(0)
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Optionally adjust output amplitude.
    audio_tensor = audio_tensor * output_gain

    # Convert to NumPy for further processing.
    audio_np = audio_tensor.squeeze(0).cpu().numpy()

    # ----- Spectral Denoising -----
    if apply_spectral_denoise and nr is not None:
        audio_np = nr.reduce_noise(y=audio_np, sr=sampling_rate, prop_decrease=spectral_denoise_prop_decrease)
        print(f"Applied spectral denoising with prop_decrease={spectral_denoise_prop_decrease}.")

    # Optionally apply de–emphasis filtering.
    if apply_deemph:
        audio_np = apply_deemphasis(audio_np)
        print("Applied de–emphasis filtering.")

    # Optionally normalize final audio.
    if normalize_final:
        audio_np = normalize_audio(audio_np)
        print("Normalized final audio amplitude.")

    # ---- Apply a Final Boost ----
    audio_np = audio_np * final_boost
    print(f"Applied final boost factor of {final_boost}.")

    # Convert back to a torch tensor with shape [1, time_samples].
    audio_tensor = torch.tensor(audio_np).unsqueeze(0)

    generated_samples = audio_tensor.shape[1]
    generated_duration = generated_samples / sampling_rate
    expected_duration = (num_time_steps * hop_size) / sampling_rate
    print(f"Expected Duration: {expected_duration:.2f}s | Generated Duration: {generated_duration:.2f}s")

    # Save the generated audio.
    torchaudio.save(output_path, audio_tensor, sampling_rate, format='wav')
    print(f"Generated audio saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', default='./cp_hifigan/generator.onnx', help='Path to the ONNX model')
    parser.add_argument('--output_path', default='./generated_audio.wav', help='Output audio file path')
    parser.add_argument('--spectrogram_path', default='./dataset/mels/0058.npy', help='Path to the mel spectrogram file')
    parser.add_argument('--sampling_rate', type=int, default=22050, help='Audio sampling rate')
    parser.add_argument('--hop_size', type=int, default=256, help='Hop size used during spectrogram generation')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference (cpu or cuda)')
    parser.add_argument('--apply_log', action='store_true', help='Apply logarithmic scaling')
    parser.add_argument('--apply_norm', action='store_true', help='Apply normalization')
    parser.add_argument('--output_gain', type=float, default=1.0, help='Output amplitude gain multiplier')
    args = parser.parse_args()

    generate_audio(
        onnx_path=args.onnx_path,
        output_path=args.output_path,
        spectrogram_path=args.spectrogram_path,
        sampling_rate=args.sampling_rate,
        hop_size=args.hop_size,
        device=args.device,
        apply_log_scale=args.apply_log,
        normalize_mel=args.apply_norm,
        output_gain=args.output_gain
    )

if __name__ == '__main__':
    main()
