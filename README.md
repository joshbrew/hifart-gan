# hifart-gan
HiFi-GAN optimized to about 40%-50% faster than the [original](https://github.com/jik876/hifi-gan) without tweaking model parameters just the train.py. Also with a fart sound library to train it to... well yeah. If it doesn't work just replace the original files and there are a few fixes GPT can make for you in train.py and meldataset.py. Follow [AddingData.md](./AddingData.md) for how to add your datasets. There are a few relpaths you may need to change.

I had to use Ubuntu 18.04 due to some package issues. Also python 3.8, the model says it supports newer but I tried 5 ubuntu versions and 5 python version and only got 1 combination to work.

For even more data: https://www.kaggle.com/datasets/alecledoux/fart-recordings-dataset

Comes with a ipynb notebook you can run on Google Colab if you upload this repo to your drive.

You need pytorch and ffmpeg and CUDA support. Adjust batch sizes for your GPU. 16-32 is fine on an RTX 3070, 128-256 on an A100. 

See [Original Readme](./Original_README.md) for more info.

Our attempt converged to 25%-30% error in 3100 (default) epochs. Took about 4 hours on an A100 (Google Colab w/ google drive copy of repo) at 4s per epoch. The models typically get trained to 500k-1M epochs for publishing. I got about 30s to 40s per epoch on my RTX 3070, better than the 60s-70s of the original code.