When adding a dataset, in this case programmed for a set of wav files, go in this order:

First off:
`pip install -r requirements.txt`
Also:
`sudo apt-get install ffmpeg`

You'll also want the CUDA dependencies with PyTorch, including torchvision and torchaudio.

> open convert.py and set the source wav file directory and target spectrogram npy file directory then run `python3 convert.py`

> run `python3 rename.py`
> run `python3 prepare.py`
> run `python3 train.py --config config_v3.json --input_wavs dataset/wavs --input_mels dataset/mels`

