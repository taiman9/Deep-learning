import numpy as np

def computeNumericalGradient(J, theta):
  """ Compute numgrad = computeNumericalGradient(J, theta)

  theta: a vector of parameters
  J: a function that outputs a real-number and the gradient.
  Calling y = J(theta)[0] will return the function value at theta. 
  """

  # Initialize numgrad with zeros
  numgrad = np.zeros(theta.size)

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Implement numerical gradient checking, and return the result in numgrad.  
  # (See Section 2.3 of the lecture notes.)
  # You should write code so that numgrad(i) is (the numerical approximation to) the 
  # partial derivative of J with respect to the i-th input argument, evaluated at theta.  
  # I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
  # respect to theta(i).
  #                
  # Hint: You will probably want to compute the elements of numgrad one at a time. 

  e = 0.0001
  p = np.zeros(theta.size)
  length = len(numgrad)
  for i in range(length):
      p[i]=e
      loss1 = J(theta+p)[0]
      loss2 = J(theta-p)[0]
      numgrad[i] = (loss1-loss2)/(2*e)
      p[i]=0



  ## ---------------------------------------------------------------

  return numgrad
