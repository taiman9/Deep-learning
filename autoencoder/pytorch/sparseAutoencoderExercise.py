import argparse
import sys

import numpy as np
from numpy.random import randn, randint
from numpy.linalg import norm
import matplotlib.pyplot as plt

import sparseAutoencoder as sAE
from displayNetwork import displayNetwork

import torch


def train():
  """Train sparse Auto-Encoder for a number of epochs.
  """
  #  In this section, we load the sets of images.
  if FLAGS.input_type == 'natural':
    from sampleNaturalImages import sampleNaturalImages
    numPatches = 10000
    mb_size = 2000
    patches = sampleNaturalImages(FLAGS.data_dir + 'images.mat', numPatches)
    learning_rate = FLAGS.learning_rate
    num_epochs = 4000
  else:
    numPatches = 20000
    mb_size = FLAGS.batch_size
    from sampleDigitImages import sampleDigitImages
    patches = sampleDigitImages(FLAGS.data_dir + 'mnist', numPatches)
    learning_rate = FLAGS.learning_rate
    num_epochs = FLAGS.num_epochs

  #print(patches.shape)

  # Initialize parameters, wrap in variables for autograd.
  W1, b1, W2, b2 = sAE.get_vars(FLAGS.visibleSize, FLAGS.hiddenSize)

  dtype = torch.FloatTensor
  optimizer = torch.optim.Adam([W1, b1, W2, b2], lr = learning_rate)
  num_mbs = numPatches // mb_size
  # Gradient descent loop.
  for epoch in range(num_epochs + 1):
    total = 0.0
    # For each minibatch.
    for mb in range(0, num_mbs):
      first = mb_size * mb
      last = mb_size * (mb + 1)
      X = torch.from_numpy(patches[:, first : last]).type(dtype)

      # Forward propagation.
      loss = sAE.cost(X, W1, b1, W2, b2, FLAGS.decay, FLAGS.rho, FLAGS.beta)

      total += loss

      # Zero the gradients.
      optimizer.zero_grad()
      
      # Use autograd to compute the backward pass.
      loss.backward()

      # Update weights.
      optimizer.step()

    print("Epoch %d: cost %f" % (epoch, total.data / num_mbs))

  return W1



## ------------------------ Main -----------------------##
if __name__ == '__main__':
  parser = argparse.ArgumentParser('Sparse AutoEncoder Exercise.')
  parser.add_argument('--input_type',
                      type=str,
                      choices = ['natural', 'digits'],
                      default='natural',
                      help = 'Type of images used for training.')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=100,
      help='Number of epochs to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=200,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='../../data/',
      help='Directory to put the training data.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  
  parser.add_argument('--visibleSize', type=int)
  parser.add_argument('--hiddenSize', type=int)
  parser.add_argument('--rho', type=float)
  parser.add_argument('--decay', type=float)
  parser.add_argument('--beta', type = float)
  if FLAGS.input_type == 'natural':
    parser.parse_args(args=['--visibleSize', str(8*8)], namespace=FLAGS)
    parser.parse_args(args=['--hiddenSize', '25'], namespace=FLAGS)
    parser.parse_args(args=['--rho', '0.01'], namespace=FLAGS)
    parser.parse_args(args=['--decay', '0.0001'], namespace=FLAGS)
    parser.parse_args(args=['--beta', '3'], namespace=FLAGS)
  else:
    parser.parse_args(args=['--visibleSize', str(28*28)], namespace=FLAGS)
    parser.parse_args(args=['--hiddenSize', '196'], namespace=FLAGS)
    parser.parse_args(args=['--rho', '0.1'], namespace=FLAGS)
    parser.parse_args(args=['--decay', '3e-3'], namespace=FLAGS)
    parser.parse_args(args=['--beta', '3'], namespace=FLAGS)
  
  np.random.seed(1)

  # Train filters.
  weights = train()

  # Display learned filters.
  displayNetwork(weights.data.numpy().T, file_name = 'weights-' + FLAGS.input_type + '.jpg')
