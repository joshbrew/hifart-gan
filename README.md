# hifart-gan
HiFi-GAN optimized to about 40%-50% faster than the [original](https://github.com/jik876/hifi-gan) without tweaking model parameters just the train.py. Also with a fart sound library to train it to... well yeah. No idea if it converges, it takes like a week on a 3070 and I don't have a spare to leave running.

For even more data: https://www.kaggle.com/datasets/alecledoux/fart-recordings-dataset

Comes with a ipynb notebook you can run on Google Colab if you upload this repo to your drive.

You need pytorch and ffmpeg and CUDA support. Adjust batch sizes for your GPU. 16-32 is fine on an RTX 3070, 128-256 on an A100. 
