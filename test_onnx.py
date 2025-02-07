import torch
import torchaudio
import argparse
import onnxruntime as ort
import numpy as np
print(torchaudio.list_audio_backends())


def generate_audio(onnx_path, output_path, num_mels=80, sampling_rate=22050, duration=3, device='cpu'):
    """Generate a 3-second audio clip using the exported ONNX model."""

    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider'])
    
    # Calculate time steps for given duration
    time_steps = (duration * sampling_rate) // 256  # Adjust hop_size if needed
    dummy_input = np.random.randn(1, num_mels, time_steps).astype(np.float32)

    # Run inference
    audio_output = session.run(None, {'mel': dummy_input})[0]  # Extract output

    # Convert to PyTorch tensor
    audio_tensor = torch.tensor(audio_output).squeeze(0)  # Shape: (1, time_samples)

    # Ensure tensor has correct shape (1, num_samples)
    if audio_tensor.ndim == 3:  
        audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension

    # Ensure it's 2D (channels, samples)
    if audio_tensor.ndim == 1:  
        audio_tensor = audio_tensor.unsqueeze(0)  # Convert to (1, time_samples)

    # Save audio
    torchaudio.save(output_path, audio_tensor, sampling_rate, format='wav')
    print(f"Generated audio saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', default='./cp_hifigan/generator.onnx', help='Path to the ONNX model')
    parser.add_argument('--output_path', default='./generated_audio.wav', help='Path to save the generated audio clip')
    parser.add_argument('--num_mels', default=80, type=int, help='Number of mel channels')
    parser.add_argument('--sampling_rate', default=22050, type=int, help='Audio sampling rate')
    parser.add_argument('--duration', default=3, type=int, help='Duration of the generated audio in seconds')
    parser.add_argument('--device', default='cpu', help='Device to run inference on (default: cpu)')
    args = parser.parse_args()

    generate_audio(args.onnx_path, args.output_path, args.num_mels, args.sampling_rate, args.duration, args.device)

if __name__ == '__main__':
    main()
