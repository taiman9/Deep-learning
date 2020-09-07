import numpy as np
from numpy.random import randn, randint

from scipy.sparse import coo_matrix


def forward(X, params):
  W1 = params['W1']
  b1 = params['b1']
  W2 = params['W2']
  b2 = params['b2']

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Compute Z2, A2, Z3, A3 (the softmax output).
  Z2 = np.dot(W1,X) + b1
  A2 = relu(Z2)
  Z3 = np.dot(W2,A2) + b2
  overflow = np.max(Z3)
  matrix_sc = Z3 - overflow
  matrix_exp = np.exp(matrix_sc)
  matrix_p = matrix_exp/np.sum(matrix_exp,axis=0)
  A3 = matrix_p 

  
  ## ----------------------------------------------------------------

  cache = {'Z2': Z2,
           'A2': A2,
           'Z3': Z3,
           'A3': A3}

  return A3, cache


def backward(X, y, params, cache, decay):
  W1 = params['W1']
  b1 = params['b1']
  W2 = params['W2']
  b2 = params['b2']

  A1 = X
  Z2 = cache['Z2']
  A2 = cache['A2']
  Z3 = cache['Z3']
  A3 = cache['A3']

  m = X.shape[1]
  groundTruth = coo_matrix((np.ones(m, dtype = np.uint8),
                            (y, np.arange(m)))).toarray()

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Compute gradients dW2, db2, dW1, db1.
  
  dZ2 = A3 - groundTruth 
  dW2 = (1 / m)*np.dot(dZ2, A2.T) + decay*W2
  db2 = (1 / m)*np.sum(dZ2, axis=1, keepdims=True)
  dZ1 = np.dot(W2.T,dZ2)
  dZ1[A2<=0] = 0
  dW1 = (1 / m)*np.dot(dZ1, A1.T) + decay*W1
  db1 = (1 / m)*np.sum(dZ1, axis=1, keepdims=True)

  
  ## ----------------------------------------------------------------

  dParams = {'dW1': dW1,
             'db1': db1,
             'dW2': dW2,
             'db2': db2}

  return dParams


def updateParams(params, dParams, learning_rate):
  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Use gradients in dParams to update params in params.

  # Retrieve each parameter from the dictionary "params"

  W1 = params['W1']
  b1 = params['b1']
  W2 = params['W2']
  b2 = params['b2']

    
  # Retrieve each gradient from the dictionary "dParams"
  dW1 = dParams['dW1']
  db1 = dParams['db1']
  dW2 = dParams['dW2']
  db2 = dParams['db2']

  W1 = W1 - learning_rate*dW1
  b1 = b1 - learning_rate*db1
  W2 = W2 - learning_rate*dW2
  b2 = b2 - learning_rate*db2

  dW1 = 0
  db1 = 0
  dW2 = 0
  db2 = 0



  params = {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2}

  return params
  ## ----------------------------------------------------------------


def cost(X, y, params, decay):
  
  W1 = params['W1']
  W2 = params['W2']

  m = X.shape[1]
  groundTruth = coo_matrix((np.ones(m, dtype = np.uint8),
                            (y, np.arange(m)))).toarray()

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Compute cost of NN model.
  a3, cache = forward(X, params)
  cost = np.multiply(groundTruth, np.log(a3))
  cost = -(np.sum(cost)/m)
  w_squared = np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2,W2)) 
  reg = 0.5*decay*w_squared
  cost= cost + reg

  
  ## ----------------------------------------------------------------

  return cost


def predict(X, params):
  W1 = params['W1']
  b1 = params['b1']
  W2 = params['W2']
  b2 = params['b2']

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Compute label predictions of the NN model.

  pred = np.zeros(X.shape[1])
  a3, cache = forward(X, params)
  pred=np.argmax(a3, axis = 0)
  
  ## ----------------------------------------------------------------

  return pred


def initParams(n_x, n_h, n_y):
  W1 = 0.01 * randn(n_h, n_x)
  b1 = np.zeros((n_h, 1))
  W2 = 0.01 * randn(n_y, n_h)
  b2 = np.zeros((n_y, 1))

  params = {'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2}

  return params


def ravelGrads(params):
  dW1 = params['dW1']
  db1 = params['db1']
  dW2 = params['dW2']
  db2 = params['db2']

  return np.hstack((dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()))


def relu(x):
  return np.maximum(0, x)


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) 


def tanh(x):
  e = np.exp(2 * x - 1)
  return (e - 1) / (e + 1)
