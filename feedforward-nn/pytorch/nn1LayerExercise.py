import argparse
import sys

import numpy as np
from numpy.random import randn, randint
from numpy.linalg import norm
import matplotlib.pyplot as plt

import nn1Layer
import utils

import torch
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser('NN with 1 Hidden Layer Exercise.')
parser.add_argument('-i', '--input_data',
                    type=str,
                    default='spiral',
                    help='Dataset: select between "spiral" and "flower".')
parser.add_argument('-d', '--debug',
                    action='store_true',
                    help='Used for gradient checking.')

FLAGS, unparsed = parser.parse_known_args()

torch.manual_seed(3)

##======================================================================
## STEP 1: Load data
#
#  In this section, we load the training instances and their labels.

if FLAGS.input_data == 'spiral':
  X, y, n_y = utils.load_spiral_dataset()
  # Set hyper-parameters.
  n_h = 100
  decay = 0.001
  learning_rate = 1
  num_epochs = 10000
elif FLAGS.input_data == 'flower':
  X, y, n_y = utils.load_flower_dataset()
  # Set hyper-parameters.
  n_h = 20
  decay = 0
  learning_rate = 0.05
  num_epochs = 20000
else:
  print('Wrong dataset specified. Select between "spiral" and "flower".')
  sys.exit(1)

n_x = X.shape[0]
m = X.shape[1]

dtype = torch.FloatTensor
ltype = torch.ByteTensor


##======================================================================
## STEP 2: Initialize parameters, set up Tensors for autograd.
#

# Randomly initialize parameters.
W1 = 0.01 * torch.randn(n_h, n_x).type(dtype)
W1.requires_grad_(True)
b1 = torch.zeros((n_h, 1)).type(dtype)
b1.requires_grad_(True)
W2 = 0.01 * torch.randn(n_y, n_h).type(dtype)
W2.requires_grad_(True)
b2 = torch.zeros((n_y, 1)).type(dtype)
b2.requires_grad_(True)

# Load data into Tensors that do not require gradients.
groundTruth = coo_matrix((np.ones(m, dtype = np.uint8),
                          (y, np.arange(m)))).toarray()
groundTruth = torch.from_numpy(groundTruth).type(dtype)
zero = torch.FloatTensor([0])
X = torch.from_numpy(X).type(dtype)


##======================================================================
## STEP 3: Gradient descent loop
#
# In this section, run gradient descent for num_epochs.
# At each epoch, first compute the loss
#     by calling  nn1Layer.cost, 
#     Print cost every 1000 epochs.
# then update params using the gradient automatically computed
#     by calling loss.backward().

## ---------- YOUR CODE HERE --------------------------------------

# Gradient descent loop.
for epoch in range(num_epochs):
  # Compute loss variable.
  loss = nn1Layer.cost(X, groundTruth, zero, W1, b1, W2, b2, decay)

  # Print cost every 1000 iterations.
  if epoch % 1000 == 0:
    print("Epoch %d: cost %f" % (epoch, loss))

  #Backward propagation.
  loss.backward()
    
  # Gradient update.
  W1.data -= learning_rate * W1.grad.data
  b1.data -= learning_rate * b1.grad.data
  W2.data -= learning_rate * W2.grad.data
  b2.data -= learning_rate * b2.grad.data
          
  # Manually zero the gradients after updating weights
  W1.grad.zero_()
  b1.grad.zero_()
  W2.grad.zero_()
  b2.grad.zero_()       
  

  
## ----------------------------------------------------------------

  

##======================================================================
## STEP 4: Test on training data
#
#  Now test your model against the training examples.
#  The array pred should contain the predictions of the NN model.

X = X.data.numpy()
W1 = W1.data.numpy()
b1 = b1.data.numpy()
W2 = W2.data.numpy()
b2 = b2.data.numpy()

pred = nn1Layer.predict(X, W1, b1, W2, b2)

acc = np.mean(y == pred)
print('Accuracy: %0.3f%%.' % (acc * 100))

# Accuracy is the proportion of correctly classified images.
# The results for our implementation were:
#
# Spiral Accuracy: 99.00%
# Flower Accuracy: 87.00%


##======================================================================
## STEP 5: Plot the decision boundary
#

utils.plot_decision_boundary(lambda x: nn1Layer.predict(x, W1, b1, W2, b2),
                             X, y)
plt.title("Neural Network")
plt.savefig(FLAGS.input_data + '-boundary.jpg')
plt.show()
