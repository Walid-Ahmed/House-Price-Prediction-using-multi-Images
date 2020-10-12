# House Price Prediction using multi Images 

This repo uses deep learning to predict the price of a house based on four images

1. Bathroom
2. Kitchen
3. Frontal
4. Bedroom

Each image is fed to a branch of convolution  network, features are extracted from each image then concatenated  and all features are fed to a fully connected NN to predict the price

The dataset is from   https://github.com/emanhamed/Houses-dataset

## Usage

python train.py --numOfEpochs  20 --width  128 --height 128 --channels 3  --normalizeFlag False --weightSharing False


<img src="https://github.com/Walid-Ahmed/House-Price-Prediction-using-multi-Images/blob/master/sampleImages/plot_acc.png"  align="middle">

## Description

The model used is as the following:

Trainable params: 3,488,129

<img src="https://github.com/Walid-Ahmed/House-Price-Prediction-using-multi-Images/blob/master/sampleImages/model.png"  align="middle">


If  model used is with weight sharing, it will be as the following:

Trainable params: 3,458,801

<img src="https://github.com/Walid-Ahmed/House-Price-Prediction-using-multi-Images/blob/master/sampleImages/modelSharedWeights.png"  align="middle">


And model summary as following:

<img src="https://github.com/Walid-Ahmed/House-Price-Prediction-using-multi-Images/blob/master/sampleImages/modelSummary_WeightSharing.png"  align="middle">

## Credits
Special thanks to [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/)   for his  [great post](https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns//) that inspired this tutourial. In his code he concetenated the four images into one image and it was worth a shot  to try using different branches for each image.



