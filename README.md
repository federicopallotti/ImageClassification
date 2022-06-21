# ImageClassification
## Dataset:
Working with a dataset of 50000 training images and 10000 test images, grouped into 10 classes.

Here are some example images from the training set:

<img width="673" alt="Schermata 2022-06-21 alle 14 55 08" src="https://user-images.githubusercontent.com/77103965/174804911-7a8b0d86-695c-4951-8b3b-2b88a6526626.png">

Implementation of a transformation pipeline in order to obtain data with mean = 0 and standard deviation = 1.
The following values for the 3 channels, on the entire dataset:

• means = [0.49139968, 0.48215841, 0.44653091]

• std. deviations = [0.24703223, 0.24348513, 0.26158784]

These values are then fed to the Normalization function, that we need for the
normalization of each input image. Here are some normalized images:

<img width="679" alt="Schermata 2022-06-21 alle 14 55 00" src="https://user-images.githubusercontent.com/77103965/174805014-71d01fb8-5432-4dbb-a74d-595db222860e.png">

## CNN
The following convolutional neural networks was implemented using rectified linear activation functions (ReLUs):

• Convolutional layer 1: 32 filters, 3 × 3. (b) Convolutional layer 2: 32 filters, 3 × 3.

• Max-pooling layer 1: 2 × 2 windows.

• Convolutional layer 3: 64 filters, 3 × 3. 

• Convolutional layer 4: 64 filters, 3 × 3. 

• Max-pooling layer 2: 2 × 2 windows. 

• Fully connected layer 1: 512 units.

• Softmax output layer.

## Training
The training pipeline was implemented, checking the training loss and accuracy every 100 steps, and keeping track of the best epoch, where the best validation accuracy was found.

## Final Validation Accuracy
The best Validation Accuracy obtained is 74.4 %, obtained at the epoch 11, with a loss of 0.7797.

<img width="704" alt="Schermata 2022-06-21 alle 15 09 09" src="https://user-images.githubusercontent.com/77103965/174806917-cfa7044e-9ea5-4f0d-a9dc-53586ac10515.png">

After the 7th epoch, approxi- mately, both the accuracies and the losses start do diverge.
This is due to overfitting of the model that performs well on the training set but not on the validation set.


