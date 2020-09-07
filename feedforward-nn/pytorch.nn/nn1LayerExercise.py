import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

dtype = torch.FloatTensor
ltype = torch.LongTensor


##======================================================================
## STEP 2: Initialize parameters, set up Variables for autograd.
#

# Load data into Variables that do not require gradients.
X = Variable(torch.from_numpy(X.T).type(dtype), requires_grad = False)
y = Variable(torch.from_numpy(y).type(ltype), requires_grad = False)


## -------------------- YOUR CODE HERE ------------------------------
# Define model to be a NN with one hidden ReLU layer and linear outputs.
# Define loss_fn to compute the cross entropy loss.
# Define the optimizer to run SGD with learning_rate and the weight decay.

model = nn.Sequential(nn.Linear(n_x, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h,n_y))

loss_fn = nn.CrossEntropyLoss() 

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

## ----------------------------------------------------------------


##======================================================================
## STEP 3: Gradient descent loop
#

## ---------- YOUR CODE HERE --------------------------------------
# In this section, run gradient descent for num_epochs.
# At each epoch:
#   - first compute the model predictions;
#   - then compute the loss between predictions and true labels;
#          print hte loss every 100 epochs;
#   - then zero de gradients through the optimizer object;
#   - then run backpropagation on the loss.

# Gradient descent loop
for epoch in range(num_epochs):

  # Compute model predictions
  y_pred = model(X)
  
  # Compute loss
  loss = loss_fn(y_pred, y)
 
  # Print cost every 1000 iterations.
  if epoch % 1000 == 0:
    print("Epoch %d: cost %f" % (epoch, loss))

  # Zero the gradients
  optimizer.zero_grad()

  # Backpropogate on loss
  loss.backward()

  #Update the parameters
  optimizer.step()
  

  
## ----------------------------------------------------------------


##======================================================================
## STEP 4: Test on training data
#
#  Now test your model against the training examples.
#  The array pred should contain the predictions of the softmax model.

pred = model(X)
#pred = F.softmax(pred, dim=0)
pred = pred.data.numpy()
pred = np.argmax(pred, axis = 1)

#print(pred[0:10])
#print(y.data.numpy()[0:10])

acc = np.mean(y.data.numpy() == pred)
print('Accuracy: %0.3f%%.' % (acc * 100))

# Accuracy is the proportion of correctly classified images
# After 200 epochs, the results for our implementation were:
#
# Spiral Accuracy: 99.00%
# Flower Accuracy: 87.00%
