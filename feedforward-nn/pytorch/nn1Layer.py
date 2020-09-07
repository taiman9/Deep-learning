import numpy as np
import torch

def cost(X, groundTruth, zero, W1, b1, W2, b2, decay):
  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Compute negative log-likelihood loss.

  m = X.data.shape[1]

  Z2 = torch.mm(W1,X) + b1
  A2 = torch.max(zero, Z2)
  Z3 = torch.mm(W2,A2) + b2
  overflow = torch.max(Z3)
  matrix_sc = Z3 - overflow
  matrix_exp = torch.exp(matrix_sc)
  matrix_p = matrix_exp/torch.sum(matrix_exp,dim=0)
    
  loss  = torch.mul(groundTruth, torch.log(matrix_p))
  loss = -(torch.sum(loss) / m)
  w_squared = torch.sum(torch.mul(W1, W1)) + torch.sum(torch.mul(W2, W2))
  reg = 0.5 *decay  * w_squared
    
  loss = loss + reg

  ## ----------------------------------------------------------------

  return loss



def predict(X, W1, b1, W2, b2):
  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Compute NN label predictions.
  Z2 = np.dot(W1,X) + b1
  A2 = np.maximum(0, Z2)
  Z3 = np.dot(W2,A2) + b2
  overflow = np.max(Z3)
  matrix_sc = Z3 - overflow
  matrix_exp = np.exp(matrix_sc)
  matrix_p = matrix_exp/np.sum(matrix_exp,axis=0)
  A3 = matrix_p 
    
  pred = np.zeros(X.shape[1])
  pred = np.argmax(A3, axis = 0)
  
  ## ----------------------------------------------------------------
  
  return pred
