import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#torch.manual_seed(2)

num_epochs = 50000


dtype = torch.FloatTensor
ltype = torch.LongTensor

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]]).type(dtype)
y = torch.tensor([[0],[1],[1],[0]]).type(dtype)


model = nn.Sequential(nn.Linear(2, 2),
                      nn.Sigmoid(),
                      nn.Linear(2, 1),
 		      nn.Sigmoid())


loss_fn = nn.MSELoss() 

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(num_epochs):

  # Compute model predictions
  y_pred = model(X)
  
  # Compute loss
  loss = loss_fn(y_pred, y)
 
  # Print cost every 1000 iterations.
  if epoch % 5000 == 0:
    print("Epoch %d: cost %f" % (epoch, loss))

  # Zero the gradients
  optimizer.zero_grad()

  # Backpropogate on loss
  loss.backward()

  #Update the parameters
  optimizer.step()
 
 
for _x, _y in zip(X, y):
 prediction = model(_x)
 print('Input:\t', list(map(int, _x)))
 if prediction >= 0.5:
   prediction = 1
 else:
   prediction = 0
 print('Pred:\t', prediction)
 print('Ouput:\t', int(_y))
 print('\n')


