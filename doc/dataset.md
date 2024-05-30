# Data

We'll be using the MNIST dataset. It consists of images of handwritten digits from 0-9. Each images are 28x28 pixels, this gives us a total of 784 pixels. Each of those pixels is just a color value between 0 and 255 (0 being black, 255 being white).

The dataset is represented as matrix. with each row of the matrix being an image, and each row will contain 784 columns.

We'll need to transpose this dataset, turning each rows into columns

We'll also split this dataset into 2 different sets: a training dataset and a testing dataset. The training dataset will have 41000 entries and the testing dataset will have 1000 entries.
