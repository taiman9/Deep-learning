===================== How to create dataset ==================
> ln -s ../../data

Create customized embeddings file and data set:
python3 data_set.py data/snli_1.0 data/GoogleNews-vectors-negative300.bin data/embedding.pkl data/snli_padding.pkl


==================== How to run the program ==================

model 1:
python rnnExercise.py 1

model 2:
python rnnExercise.py 2

model 3:
python rnnExercise.py 3


=========================== Some tips ========================

1. Pytorch RNN code on GPU is much faster than tensorflow, try to use GPU.


2. You need to extend the torch.nn.Module to fulfill the assignment, when you
   define variable manually, define them with torch.nn.Parameter, otherwise,
   the variables can not be registered into the module, hence can not be trained.


3. The LSTM in pytorch does not initialize the cell states and hidden states
   by default, you need to manually set them.


4. Try to use torch.repeat or torch.expand when you need to extend a tensor
   over some dimension. The difference between torch.repeat and torch.expand
   is that torch.expand does not allocate memory for the extended elements.

5. Try use different number of linear layers between LSTM and softmax layer,
   different location to insert dropout layer, different activation functions 
   for linear layers, and the distribution of the initial values of the manually 
   created variables. All these may affect the accuracy.

