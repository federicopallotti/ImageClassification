# ObjectDetection
## Dataset:
We are currently working with a dataset of 50000 training images and 10000 test images, grouped into 10 classes.
Here are some example images from the training set:

<img width="673" alt="Schermata 2022-06-21 alle 14 55 08" src="https://user-images.githubusercontent.com/77103965/174804911-7a8b0d86-695c-4951-8b3b-2b88a6526626.png">

Here we create a transformation pipeline in order to obtain data with mean = 0 and standard deviation = 1.
We found the following values for the 3 channels, on the entire dataset:
• means = [0.49139968, 0.48215841, 0.44653091]
• std. deviations = [0.24703223, 0.24348513, 0.26158784]
These values are then fed to the Normalization function, that we need for the
normalization of each input image. Here are some normalized images:

<img width="679" alt="Schermata 2022-06-21 alle 14 55 00" src="https://user-images.githubusercontent.com/77103965/174805014-71d01fb8-5432-4dbb-a74d-595db222860e.png">

## CNN
