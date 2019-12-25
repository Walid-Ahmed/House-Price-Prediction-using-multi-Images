# multiImageInputRegression

This repo uses deep learning to predict the price of a house beased on four images, each image is fed to a branch of convloution netweork, features are extracted from each image then concetenated and all features are fed to a fully connected NN to predict the price


## Usage

python train.py --numOfEpochs  20 --width  128 --height 128 --channels 3  --normalizeFlag False --weightSharing False

## Description

The model used is as the following:

<img src="https://github.com/Walid-Ahmed/multiImageInputRegression/blob/master/model.png"  align="middle">
