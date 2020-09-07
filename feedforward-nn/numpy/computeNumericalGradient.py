import numpy as np

def computeNumericalGradient(J, params):
  """ Compute numgrad = computeNumericalGradient(J, params)

  params: a dictionary of parameters
  J: a function that outputs a real-number and the gradient.
  Calling y = J(params) will return the function value at params. 
  """

  W1 = params['W1']
  b1 = params['b1']
  W2 = params['W2']
  b2 = params['b2']


  # Initialize numgrad with zeros.
  numd_W1 = np.zeros(W1.shape)
  numd_b1 = np.zeros(b1.shape)
  numd_W2 = np.zeros(W2.shape)
  numd_b2 = np.zeros(b2.shape)

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Implement numerical gradient checking, and return the result in numgrad.  
  epsilon = 1e-4

  W1_p=np.array(W1,dtype=np.float)
  W1_n=np.array(W1,dtype=np.float)


  for i in range(numd_W1.shape[0]):
    for j in range(numd_W1.shape[1]): 
        
        W1_p[i][j] = W1[i][j] + epsilon
        W1_n[i][j] = W1[i][j] - epsilon 

        params = {"W1": W1_p,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

        above = J(params)

        params = {"W1": W1_n,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

        below = J(params)
    
        numd_W1[i][j]= (above - below)/(2*epsilon)

        W1_p[i][j] = W1[i][j]     
        W1_n[i][j] = W1[i][j]


  W2_p=np.array(W2,dtype=np.float)
  W2_n=np.array(W2,dtype=np.float) 

  for i in range(numd_W2.shape[0]):
    for j in range(numd_W2.shape[1]): 
        
        W2_p[i][j] = W2[i][j] + epsilon
        W2_n[i][j] = W2[i][j] - epsilon 
    
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2_p,
                  "b2": b2}

        above = J(params)

        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2_n,
                  "b2": b2}

        below = J(params)

        numd_W2[i][j]= (above - below)/(2*epsilon)

        W2_p[i][j] = W2[i][j]     
        W2_n[i][j] = W2[i][j]

  
  b1_p=np.array(b1,dtype=np.float)
  b1_n=np.array(b1,dtype=np.float)

  for i in range(numd_b1.shape[0]):
        
        b1_p[i] = b1[i] + epsilon
        b1_n[i] = b1[i] - epsilon 

        params = {"W1": W1,
                  "b1": b1_p,
                  "W2": W2,
                  "b2": b2}

        above = J(params)

        params = {"W1": W1,
                  "b1": b1_n,
                  "W2": W2,
                  "b2": b2}

        below = J(params)

        numd_b1[i] = (above -below)/(2*epsilon)

        b1_p[i] = b1[i]     
        b1_n[i] = b1[i]

  
  b2_p=np.array(b2,dtype=np.float)
  b2_n=np.array(b2,dtype=np.float)

  for i in range(numd_b2.shape[0]):
        
        b2_p[i] = b2[i] + epsilon
        b2_n[i] = b2[i] - epsilon 

        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2_p}

        above = J(params)

        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2_n}

        below = J(params)
    
        numd_b2[i] = (above -below)/(2*epsilon)

        b2_p[i] = b2[i]     
        b2_n[i] = b2[i]


  ## ---------------------------------------------------------------

  num_grad = {'dW1': numd_W1, 'db1': numd_b1, 'dW2': numd_W2, 'db2': numd_b2}

  return num_grad
