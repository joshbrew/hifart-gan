When adding a dataset, in this case programmed for a set of wav files, go in this order:

First off:

`pip install -r requirements.txt`

Also on linux:

`sudo apt-get install ffmpeg`

For windows you can download it and add the directory to PATH

You'll also want the CUDA dependencies with PyTorch, including torchvision and torchaudio.

> Place your wav audio dataset in ./dataset/wavs
> run `python reformat_wavs.py` if not using 16 bit PWM audio
> run `python convert_to_mels.py`  

> run `python3 rename.py`
> run `python3 prepare.py`
> run `python3 train.py --config config_v3.json --input_wavs dataset/wavs --input_mels dataset/mels`

