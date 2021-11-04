import numpy as np
from torchvision import datasets, transforms

## ---------------------------------------------------------------
def sampleDigitImages(input_data_dir, numsamples):
  """Returns 20000 random images (28x28) from the MNIST dataset.
  """

  # Get the sets of images for training and test on MNIST.
  
  train_mnist = datasets.MNIST(input_data_dir, train=True, download=False)
  test_mnist = datasets.MNIST(input_data_dir, train=False, download=False)

  # The original images are matrices of pixels, transform them into
  # vectors of 28 * 28 pixels.
  train_images = train_mnist.data.view(train_mnist.data.shape[0], -1)
  test_images = test_mnist.data.view(test_mnist.data.shape[0], -1)

  # The original pixel values are between [0, 255], normalize them
  # to be between [0, 1].
  images = np.hstack((train_images.numpy().T / 255,
                      test_images.numpy().T / 255))

  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Fill in the variable called "samples" using data 
  #  from MNIST:
  #    -- return 'numsamples' random different samples from the 70,000 images.
  sampleIdx = np.random.choice(images.shape[1],numsamples)
  samples = images[:,sampleIdx]







  ## ---------------------------------------------------------------
  # For the autoencoder to work well we need to normalize the data
  # Specifically, since the output of the network is bounded between [0,1]
  # (due to the sigmoid activation function), we have to make sure 
  # the range of pixel values is also bounded between [0,1]
  
  # samples = normalizeData(samples);

  return samples


## ---------------------------------------------------------------
def normalizeData(patches):
  """Squash data to [0.1, 0.9] since we use sigmoid as the activation
  function in the output layer
  """
  
  # Remove DC (mean of images). 
  patches = patches - np.mean(patches, axis = 0)

  # Truncate to +/-3 standard deviations and scale to -1 to 1
  pstd = 3 * np.std(patches)
  patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd

  # Rescale from [-1,1] to [0.1,0.9]
  patches = (patches + 1) * 0.4 + 0.1

  return patches
