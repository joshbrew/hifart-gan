import torch
import os
import argparse
import json
from models import Generator
from env import AttrDict  # Ensure you have this utility for loading config

def load_config(config_path):
    """Load model hyperparameters from the JSON config file."""
    with open(config_path, "r") as f:
        config_data = json.load(f)
    return AttrDict(config_data)  # Convert JSON into a dict-like object

def export_onnx_model(backup_path, onnx_path, config_path, device='cpu'):
    """ Load a PyTorch model from a checkpoint and export it to ONNX format. """

    # Load the model configuration
    config = load_config(config_path)
    
    # Load the generator model with the full config
    model = Generator(config).to(device)
    
    # Load the saved model weights
    state_dict = torch.load(backup_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dummy input for ONNX export
    dummy_input = torch.randn(1, config.num_mels, 100, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['mel'],
        output_names=['audio'],
        dynamic_axes={
            'mel': {0: 'batch_size', 2: 'time_steps'},
            'audio': {0: 'batch_size', 1: 'time_steps'}
        }
    )
    print(f"ONNX model exported to {onnx_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup_path', default='./cp_hifigan/generator_backup.pt', help='Path to the backed-up .pt file')
    parser.add_argument('--onnx_path', default='./cp_hifigan/generator.onnx', help='Path to save the ONNX model')
    parser.add_argument('--config_path', default='./config_v3.json', help='Path to the model configuration JSON file')
    parser.add_argument('--device', default='cpu', help='Device to load the model onto (default: cpu)')
    args = parser.parse_args()
    
    export_onnx_model(args.backup_path, args.onnx_path, args.config_path, args.device)

if __name__ == '__main__':
    main()
