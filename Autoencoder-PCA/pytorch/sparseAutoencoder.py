import math
import numpy as np
import torch


def get_vars(visible_size, hidden_size):
  """Initialize parameters, set up Tensors for autograd.
  """
  # We'll choose weights uniformly from the interval [-r, r].
  r  = math.sqrt(6) / math.sqrt(hidden_size + visible_size + 1)
  dtype = torch.FloatTensor
  
  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Initialize W1 and W2 using Glorot uniform scheme, b1 and b2 with zeros.

  W1 = torch.nn.init.xavier_uniform_(torch.empty(hidden_size,visible_size)).type(dtype)
  W1.requires_grad_(True)
  b1 = torch.zeros((hidden_size, 1)).type(dtype)
  b1.requires_grad_(True)
  W2 = torch.nn.init.xavier_uniform_(torch.empty(visible_size,hidden_size)).type(dtype)
  W2.requires_grad_(True)
  b2 = torch.zeros((visible_size, 1)).type(dtype)
  b2.requires_grad_(True)
  
  ## ----------------------------------------------------------------

  return W1, b1, W2, b2


def cost(X, W1, b1, W2, b2, decay, rho, beta):
  """Compute the sparseAE cost on input images X.
  """
  ## ---------- YOUR CODE HERE --------------------------------------

  cost = 0.0
  
  # Define Inference Operations
  a2 = sigmoid(torch.add(torch.mm(W1,X),b1))
  y = sigmoid(torch.add(torch.mm(W2,a2),b2))
  rho_hat = torch.mean(a2,1)

  # Loss computations
  errorCost = torch.mean(0.5*torch.sum(torch.pow(y-X,2),dim=0))
  weightCost = 0.5*decay*(torch.sum(torch.pow(W1,2))+torch.sum(torch.pow(W2,2)))
  sparseCost = beta*(torch.sum(rho*torch.log(rho/rho_hat)+(1.-rho)*torch.log((1.-rho)/(1.-rho_hat))))
  cost = errorCost + weightCost + sparseCost




  ## ----------------------------------------------------------------

  return cost


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + torch.exp(-x))
    return s
