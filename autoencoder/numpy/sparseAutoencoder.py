import numpy as np
#np.seterr(divide='ignore', invalid='ignore')

def sparseAutoencoderCost(theta, visibleSize, hiddenSize, decay,
                          rho, beta, data):
  """Compute cost and gradient for the Sparse AutoEncoder.

    Args:
      visibleSize: the number of input units (probably 64) 
      hiddenSize: the number of hidden units (probably 25) 
      decay: weight decay parameter
      rho: the desired average activation \rho for the hidden units
      beta: weight of sparsity penalty term
      data: The 64x10000 matrix containing the training data.
            So, data(:,i) is the i-th training example. 
  """
  
  # The input theta is a vector (because L-BFGS expects the parameters to be a vector). 
  # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
  # follows the notation convention of the lecture notes. 

  W1 = np.reshape(theta[: hiddenSize * visibleSize],
                  (hiddenSize, visibleSize))
  W2 = np.reshape(theta[hiddenSize * visibleSize: 2 * hiddenSize * visibleSize], 
                  (visibleSize, hiddenSize))
  b1 = theta[2 * hiddenSize * visibleSize: 2 * hiddenSize * visibleSize + hiddenSize]
  b2 = theta[2 * hiddenSize * visibleSize + hiddenSize:]

  # Cost and gradient variables (your code needs to compute these values). 
  # Here, we initialize them to zeros. 
  #cost = 0
  W1grad = np.zeros(W1.shape) 
  W2grad = np.zeros(W2.shape)
  b1grad = np.zeros(b1.shape) 
  b2grad = np.zeros(b2.shape)

  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
  #                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
  #
  # W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
  # Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
  # as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
  # respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
  # with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
  # [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] (and similarly for W2grad, b1grad, b2grad).
  # 
  # Stated differently, if we were using batch gradient descent to optimize the parameters,
  # the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
  # 
  z2 = np.dot(W1,data)+np.reshape(b1,(hiddenSize,1))
  a2 = sigmoid(z2)
  z3 = np.dot(W2,a2)+np.reshape(b2,(visibleSize,1))
  y = sigmoid(z3)
  rho_hat = np.reshape(np.sum(a2,axis=1)/data.shape[1],(hiddenSize,1))
  cost = 0.5*np.sum((y-data)**2)/data.shape[1]+(decay/2)*(np.sum(W1**2)+np.sum(W2**2))+beta*np.sum(klterm(rho,rho_hat))
  delta3 = (y-data)*fprime(y)
  W2grad = np.dot(delta3,a2.T)/data.shape[1]+decay*W2
  b2grad = np.sum(delta3,axis=1)/data.shape[1]
  delta2 = (np.dot(W2.T,delta3)+beta*(-(rho/rho_hat)+((1-rho)/(1-rho_hat))))*fprime(a2)
  W1grad = np.dot(delta2,data.T)/data.shape[1]+decay*W1
  b1grad = np.sum(delta2,axis=1)/data.shape[1]



  #-------------------------------------------------------------------
  # After computing the cost and gradient, we will convert the gradients back
  # to a vector format (suitable for minFunc).  Specifically, we will unroll
  # your gradient matrices into a vector.
  grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

  return cost, grad

#-------------------------------------------------------------------
def fprime(a):
  return a * (1 - a)


#-------------------------------------------------------------------
def klterm(x, y):
  return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


#-------------------------------------------------------------------
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
