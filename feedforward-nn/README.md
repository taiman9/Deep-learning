# Feed-forward neural network using NumPy and PyTorch for non-linear classification

In this project, I implement 3 versions of a neural network with one hidden layer and a softmax output layer
using (1) NumPy, (2) PyTorch with autograd, and (3) PyTorch with the torch.nn and
torch.optim modules. I evaluate the 3 implementations on the same two 2D non-linear classification tasks. Implementation details are given below and can also be viewed in the **Implementation** section in the *feedforward-nn.pdf* file.

# Implementation 
Implement 3 versions of a neural network with one hidden layer and a softmax output layer,
using (1)NumPy, (2)PyTorch with autograd, and (3) PyTorch with the torch.nn and
torch.optim modules. Evaluate the 3 implementations on the same two 2D non-linear clas-
sification tasks: flower and spiral. Starter code and functions for generating the datasets
are available athttp://ace.cs.ohio.edu/~razvan/courses/dl6890/hw/hw02.zip. The
provided code also displays and saves images of the datasets and the trained model’s deci-
sion boundaries. Make sure that you organize your code in folders as shown in the table
below. Write code only in the Python files indicated in bold.

## NumPy Implementation
Forward Propagation: You will need to write code for the forward() function,
which computes and returns the softmax outputs in A3. Use ReLU on the hidden
layer, and also use a separate bias vector for the softmax layer. The function also
returns a cache with the A and Z values for the hidden and output layers.
<pre>
  dl6890/
    hw02/
      code/
        numpy/
          <b>
          nn1Layer.py
          computeNumericalGradient.py
          output.txt
          </b>
          nn1LayerExercise.py
          utils.py
          flower-boundary.jpg
          spiral-boundary.jpg
        pytorch/
          <b>
          nn1Layer.py
          nn1LayerExercise.py
          output.txt
          </b>
          utils.py
          flower-boundary.jpg
          spiral-boundary.jpg
        pytorch.nn/
          <b>
          nn1LayerExercise.py
          output.txt
          </b>
          utils.py
          flower-boundary.jpg
          spiral-boundary.jpg
        p3/
          <b>
          p3.py
          output.txt
          </b>
        p5/
          <b>
          p5.py
          computeNumericalGradient.py
          output.txt
          </b>
</pre>

**Backpropagation:** You will implement this in the backward() function, by minimizing the average loss 
on all the training examples in X, plus an L2 regularization term weighted by the decay hyper-parameter.

**Cost:** Compute the cost (average loss + L2 term) by first running forward propagation
to compute to softmax outputs.

**Predictions:** Compute model predictions in the cost() function.

**Vectorization:** It is important to vectorize your code so that it runs quickly.

**Overflow:** Make sure that you prevent overflow when computing the softmax probabilities.

**Numerical gradient:** Once you implemented the cost and the gradient in nn1Layer.py,
implement code for computing the gradient numerically in computeNumericalGradient.py.

**Gradient checking:** Use computeNumericalGradient.py to make sure that your
backward() function is computing gradients correctly. This is done by running the
main program in Debug mode, i.e. python3 nn1LayerExercise.py --debug.

(a) When doing gradient checking on a network that uses ReLU, the numerical gra-
dient is likely to be very different from the analytical gradient. Explain why.

(b) Only when doing gradient checking, replace ReLU with an arbitrary activation
(e.g. sigmoid) that is differentiable.

The norm of the difference between the numerical gradient and your analytical gradient
should be small, on the order of 10−^9.

## PyTorch Implementation

You will need to write code for the following:

**Cost:** Compute the cost = average negative log-likelihood + L2 regularization term.
Compute the cost yourself, i.e. do not use specialized PyTorch functions. In particular,
do not use functions from PyTorch (e.g. from the torch.nn module) that compute the
cross entropy loss.

**Predictions:** Use the trained model to compute labels for the training examples.
Use the NumPy array representation of the parameters, as created for you before the
nn1Layer.predict() call.

## PyTorch.NN Implementation

You will need to write code for the following:

**Model, Loss, Optimizer:** Define the model to be a NN with one hidden ReLU
layer and linear outputs. You may want to use the torch.nn.Sequential container.
Define the loss function to compute the cross-entropy – see loss functions defined in
the pytorch.nn module. Define the optimizer to run SGD with the specified learning
rate and weight decay.

**Gradient descent loop:** Write code for running gradient descent for num_epochs.
At each epoch, you will compute the model predictions using the model above, com-
pute the loss between predictions and true labels using the loss function above, print
the loss every 1000 epochs, zero de gradients through the optimizer object, then run
backpropagation on the loss object.

**Predictions:** Use the trained model to compute labels for the training examples.

## Theory Verification 

1. Verify experimentally only the positive conclusions that you reached for theory problem 3
(in *feedforward-nn.pdf*). Write your code in NumPy, PyTorch withautograd, or PyTorch with 
torch.nn and torch.optim.

2. Implement in NumPy the gradient formula that you derived for problem 5 (in *feedforward-nn.pdf*). 
Set all the variables to random values. Check the value of the analytical gradient against the
numerical gradient and the gradient computed through autograd in PyTorch.

## Bonus

Modify the data generation functions to create examples that have only two labels and write a
second version of the assignment that implements logistic regression for binary classification.

The screen output produced when running the code should be redirected to (saved into) the 
output.txt files.
