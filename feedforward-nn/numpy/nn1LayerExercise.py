## Neural Nework with Softmax Output Exercise 

import argparse
import sys

import numpy as np
from numpy.random import randn, randint
from numpy.linalg import norm
import matplotlib.pyplot as plt

import nn1Layer
from computeNumericalGradient import computeNumericalGradient

import utils

parser = argparse.ArgumentParser('NN with 1 Hidden Layer Exercise.')
parser.add_argument('-i', '--input_data',
                    type=str,
                    default='spiral',
                    help='Dataset: select between "spiral" and "flower".')
parser.add_argument('-d', '--debug',
                    action='store_true',
                    help='Used for gradient checking.')

FLAGS, unparsed = parser.parse_known_args()

np.random.seed(1)

##======================================================================
## STEP 1: Loading data
#
#  In this section, we load the training examples and their labels.

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking. 
# Here, we create a synthetic dataset using random data for testing.
if FLAGS.debug:
  X, y, n_y = randn(8, 100), randint(0, 2, 100, dtype = np.uint8), 2
  n_h = 5
  decay = 1e-3
elif FLAGS.input_data == 'spiral':
  # Load data.
  X, y, n_y = utils.load_spiral_dataset()
  # Set hyper-parameters.
  n_h = 100
  decay = 0.001
  learning_rate = 1
  num_epochs = 10000
elif FLAGS.input_data == 'flower':
  # Load data.
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

# Randomly initialize parameters
params = nn1Layer.initParams(n_x, n_h, n_y)


##======================================================================
## STEP 2: Gradient checking
#
#  As with any learning algorithm, you should always check that your
#  gradients are correct before learning the parameters.
#

if FLAGS.debug:
  a3, cache = nn1Layer.forward(X, params)
  dParams = nn1Layer.backward(X, y, params, cache, decay)

  dNumParams = computeNumericalGradient(lambda p: nn1Layer.cost(X, y, p, decay), params)

  rdParams = nn1Layer.ravelGrads(dParams)
  rdnParams = nn1Layer.ravelGrads(dNumParams)

  # Use this to visually compare the gradients side by side.
  print(rdnParams.shape)
  print(rdParams.shape)
  print(np.stack((rdnParams, rdParams)).T)

  # Compare numerically computed gradients with those computed analytically.
  diff = norm(rdnParams - rdParams) / norm(rdnParams + rdParams)
  print(diff)
  sys.exit(0)
  # The difference should be small. 
  # In our implementation, these values are usually less than 1e-7.

                                    
##======================================================================
## STEP 3: Learning parameters using Gradient Descent.
#
#  Once you have verified that your gradients are correct, you can start 
#  training your neural network code using gradient descent.


# Gradient descent loop.
for epoch in range(num_epochs):
  # Forward propagation.
  a3, cache = nn1Layer.forward(X, params)

  # Backward propagation.
  dParams = nn1Layer.backward(X, y, params, cache, decay)
  
  # Gradient update.
  params = nn1Layer.updateParams(params, dParams, learning_rate)

  # Print cost every 1000 iterations.
  if epoch % 1000 == 0:
    print("Epoch %d: cost %f" % (epoch, nn1Layer.cost(X, y, params, decay)))


##======================================================================
## STEP 4: Testing on training data
#
#  You should now test your model against the training examples.
#  To do this, you will first need to write softmaxPredict
#  (in softmax.py), which should return predictions
#  given a softmax model and the input data.

pred = nn1Layer.predict(X, params)

acc = np.mean(y == pred)
print('Accuracy: %0.3f%%.' % (acc * 100))

# Accuracy is the proportion of correctly classified images
# After 200 epochs, the results for our implementation were:
#
# Spiral Accuracy: 98.67%
# Flower Accuracy: 86.50%

##======================================================================
## STEP 5: Plot the decision boundary
#
utils.plot_decision_boundary(lambda x: nn1Layer.predict(x, params), X, y)
plt.title("Neural Network")
plt.savefig(FLAGS.input_data + '-boundary.jpg')
plt.show()
