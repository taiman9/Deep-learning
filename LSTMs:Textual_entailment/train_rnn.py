import numpy
import sys
import time
import torch
import torch.nn as NN
import torch.optim as OPT
import torch.nn.functional as F
from torch.autograd import Variable
# =============================================================================
# Model 1: Train one LSTM network to process both the premise and the hypothesis.
##
This model uses the same LSTM network to create vector representations
# for the premise and the hypothesis, then uses the concatenation of the
# two vector representations as the representation of the sentence pair
# to be used as input to a softmax layer.
##
lstm_size: the size of the LSTM cell.
# hidden_size: the size of the fully connected layers.
# drop_rate: Dropout rate.
# beta: the L2 regularizer parameter for the fully connected layers.
# rep_1: the matrix of word embeddings for the premise sentence.
# len_1: the true length of the premise sentence.
# mask_1: binary vector specifying true words (1) and dummy words used for padding (0).
# rep_2: the matrix of word embeddings for the hypothesis sentence.
# len_2: the true length of the hypothesis sentence.
#
class Model_1(NN.Module):
	def __init__(self, use_gpu, lstm_size, hidden_size, drop_out, beta, embedding, class_num):
		super(Model_1, self).__init__()
		numpy.random.seed(2)
		torch.manual_seed(2)
		# Set tensor type when using GPU
		if use_gpu:
			self.float_type = torch.cuda.FloatTensor
			self.long_type = torch.cuda.LongTensor
			torch.cuda.manual_seed_all(2)
			# Set tensor type when using CPU
		else:
			self.float_type = torch.FloatTensor
			self.long_type = torch.LongTensor
		# Define parameters for model
		self.lstm_size = lstm_size
		self.hidden_size = hidden_size
		self.embedding = embedding
		feature_size = self.embedding.weight.size()[1]
		self.drop_out = drop_out
		# The LSTM: input size, lstm size, num of layer
		self.lstm = NN.LSTM(feature_size, lstm_size, 1)
		# The fully connectedy layers
		self.linear1 = NN.Linear(lstm_size + lstm_size, hidden_size)
		self.linear2 = NN.Linear(hidden_size, hidden_size)
		self.linear3 = NN.Linear(hidden_size, hidden_size)
		# The fully connectedy layer for softmax
		self.linear4 = NN.Linear(hidden_size, class_num)
		# init hidden stats and cell states of LSTM
	def init_hidden(self, batch_size):
		return (Variable(torch.zeros(1, batch_size, self.lstm_size),
				requires_grad=False).type(self.float_type),
				Variable(torch.zeros(1, batch_size, self.lstm_size),
				requires_grad=False).type(self.float_type))
	# forward process
	def forward(self, rep1, len1, mask1, rep2, len2):
		rep = torch.cat((rep1, rep2), 0)
		length = torch.cat((len1, len2), 0)
		# Representation for input sentences
		batch_size = rep1.size()[0]
		sents = self.embedding(rep)
		# (sequence length * batch size * feature size)
		sents = sents.transpose(1, 0)
		# Initialial hidden states and cell states
		hidden = self.init_hidden(batch_size + batch_size)
		# Ouput of LSTM: sequence (length x mini batch x lstm size)
		lstm_outs, hidden = self.lstm(sents, hidden)
		# (batch size * sequence length * feature size)
		lstm_outs = lstm_outs.transpose(1, 0)
		# Get the valid output by the real length of the input sentences
		length = (length-1).view(-1, 1, 1).expand(lstm_outs.size(0), 1, lstm_outs.size(2))
		lstm_out = torch.gather(lstm_outs, 1, length)
		lstm_out = lstm_out.view(lstm_out.size(0), -1)
		# split representation to premise and hyphothesis representation
		(lstm_1_out, lstm_2_out) = torch.split(lstm_out, batch_size)
		# Concatenate premise and hypothesis representations
		lstm_1_out = F.dropout(lstm_1_out, p=self.drop_out)
		lstm_2_out = F.dropout(lstm_2_out, p=self.drop_out)
		lstm_out = torch.cat((lstm_1_out, lstm_2_out), 1)
		# Output of fully connected layers
		fc_out = F.dropout(F.tanh(self.linear1(lstm_out)), p=self.drop_out)
		fc_out = F.dropout(F.tanh(self.linear2(fc_out)), p=self.drop_out)
		fc_out = F.dropout(F.tanh(self.linear3(fc_out)), p=self.drop_out)
		# Output of Softmax
		fc_out = self.linear4(fc_out)
		return F.log_softmax(fc_out)

# =============================================================================
# Model 2: Conditional encoding.
##
This model used two LSTM networks, one for the premise, one for the
# hypothesis. The initial cell state of the hypothesis LSTM is set to be
# the last cell state of the premise LSTM. The last output of the
# hypothesis LSTM is used as the representation of the sentence pair.
##
lstm_size: the size of the LSTM cell.
# hidden_size: the size of the fully connected layers.
# drop_rate: Dropout rate.
# beta: the L2 regularizer parameter for the fully connected layers.
# rep_1: the matrix of word embeddings for the premise sentence.
# len_1: the true length of the premise sentence.
# mask_1: binary vector specifying true words (1) and dummy words used for padding (0).
# rep_2: the matrix of word embeddings for the hypothesis sentence.
# len_2: the true length of the hypothesis sentence.

class Model_2(NN.Module):
	def __init__(self, use_gpu, lstm_size, hidden_size, drop_out, beta, embedding, class_num):
		super(Model_2, self).__init__()
		numpy.random.seed(2)
		torch.manual_seed(2)
		# Set tensor type when using GPU
		if use_gpu:
			self.float_type = torch.cuda.FloatTensor
			self.long_type = torch.cuda.LongTensor
			torch.cuda.manual_seed_all(2)
		# Set tensor type when using CPU
		else:
			self.float_type = torch.FloatTensor
			self.long_type = torch.LongTensor
		# Define parameters for model
		self.lstm_size = lstm_size
		self.hidden_size = hidden_size
		self.embedding = embedding
		feature_size = self.embedding.weight.size()[1]
		self.drop_out = drop_out
		# The LSTMs:
		# lstm1: premise; lstm2: hypothesis
		self.lstm1 = NN.LSTMCell(feature_size, lstm_size)
		self.lstm2 = NN.LSTM(feature_size, lstm_size, 1)
		# The fully connectedy layers
		self.linear1 = NN.Linear(lstm_size, hidden_size)
		self.linear2 = NN.Linear(hidden_size, hidden_size)
		self.linear3 = NN.Linear(hidden_size, hidden_size)
		# The fully connectedy layer for softmax
		self.linear4 = NN.Linear(hidden_size, class_num)
		# Initialize the hidden states and cell states of LSTM
	def init_hidden(self, batch_size):
		return (torch.zeros(1, batch_size, self.lstm_size).type(self.float_type),
				torch.zeros(1, batch_size, self.lstm_size).type(self.float_type))
	# Forward process
	def forward(self, rep1, len1, mask1, rep2, len2):
		# Set batch size
		batch_size = rep1.size()[0]
		max_seq_len = rep1.size()[1]
		# Representation of input sentences
		sent1 = self.embedding(rep1)
		sent2 = self.embedding(rep2)
		# Transform sentences representations to:
		# (sequence length * batch size * feqture size)
		sent1 = sent1.transpose(1, 0)
		sent2 = sent2.transpose(1, 0)
		# First LSTM: (sequence lenght * mini batch * lstm size)
		hidden = self.init_hidden(batch_size)
		states = (hidden[0].view(batch_size, -1), hidden[1].view(batch_size, -1))
		lstm_outs = []
		cell_states = []
		for i in range(max_seq_len):
			states = self.lstm1(sent1[i], states)
			lstm_outs.append(states[0].view(1, batch_size, -1))
			cell_states.append(states[1].view(1, batch_size, -1))
		lstm_outs = torch.cat(lstm_outs, 0)
		cell_states = torch.cat(cell_states, 0)
		cell_states = cell_states.transpose(1, 0)
		length = (len1-1).view(-1, 1, 1).expand(cell_states.size(0), 1, cell_states.size(2))
		hidden_1 = torch.gather(cell_states, 1, length)
		# Last hidden states and cell states of the first LSTM
		hidden_1 = hidden_1.transpose(1, 0)
		# Second LSTM: (sequence lenght * mini batch * lstm size)
		# Initialize the cell states of with the last cell states of the first LSTM
		hidden = self.init_hidden(batch_size)
		hidden = (hidden[0], hidden_1)
		lstm_outs, hidden = self.lstm2(sent2, hidden)
		lstm_outs = lstm_outs.transpose(1, 0)
		length = (len2-1).view(-1, 1, 1).expand(lstm_outs.size(0), 1, lstm_outs.size(2))
		lstm_out = torch.gather(lstm_outs, 1, length)
		lstm_out = lstm_out.view(lstm_out.size(0), -1)
		lstm_out = F.dropout(lstm_out, p=self.drop_out)
		# Fully connected layers
		fc_out = F.dropout(F.tanh(self.linear1(lstm_out)), p=self.drop_out)
		fc_out = F.dropout(F.tanh(self.linear2(fc_out)), p=self.drop_out)
		fc_out = F.dropout(F.tanh(self.linear3(fc_out)), p=self.drop_out)
		# Fully connected layer for softmax
		fc_out = self.linear4(fc_out)
		return F.log_softmax(fc_out)

# =============================================================================
# Model 3: Use attention for last LSTM output of hypothesis.
##
This model use an attention mechanism, where the attention weights
# are computed between the last output of the hypothesis LSTM and all
# the outputs of the premise LTSM.
##
lstm_size: the size of the LSTM cell.
# hidden_size: the size of the fully connected layers.
# drop_rate: Dropout rate.
# beta: the L2 regularizer parameter for the fully connected layers.
# rep_1: the matrix of word embeddings for the premise sentence.
# len_1: the true length of the premise sentence.
# mask_1: binary vector specifying true words (1) and dummy words used for padding (0).
# rep_2: the matrix of word embeddings for the hypothesis sentence.
# len_2: the true length of the hypothesis sentence.

class Model_3(NN.Module):
	def __init__(self, use_gpu, lstm_size, hidden_size, drop_out, beta, embedding, class_num):
		super(Model_3, self).__init__()
		numpy.random.seed(2)
		torch.manual_seed(2)
		# Set tensor type when using GPU
		if use_gpu:
			self.float_type = torch.cuda.FloatTensor
			self.long_type = torch.cuda.LongTensor
			torch.cuda.manual_seed_all(2)
		# Set tensor type when using CPU
		else:
			self.float_type = torch.FloatTensor
			self.long_type = torch.LongTensor
		# Define parameters for model
		self.lstm_size = lstm_size
		self.hidden_size = hidden_size
		self.embedding = embedding
		feature_size = self.embedding.weight.size()[1]
		self.drop_out = drop_out
		# The LSTMs: lstm1 - premise; lstm2 - hypothesis
		self.lstm1 = NN.LSTMCell(feature_size, lstm_size)
		self.lstm2 = NN.LSTM(feature_size, lstm_size, 1)
		# The fully connectedy layers
		self.linear1 = NN.Linear(lstm_size, hidden_size)
		self.linear2 = NN.Linear(hidden_size, hidden_size)
		self.linear3 = NN.Linear(hidden_size, hidden_size)
		# The fully connectedy layer for softmax
		self.linear4 = NN.Linear(hidden_size, class_num)
		# transformation of the states
		u_min = -0.5
		u_max = 0.5
		self.Wy = NN.Parameter(torch.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
		self.Wh = NN.Parameter(torch.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
		self.Wp = NN.Parameter(torch.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
		self.Wx = NN.Parameter(torch.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
		self.aW = NN.Parameter(torch.Tensor(1, lstm_size).uniform_(u_min, u_max))
		
	# Initialize hidden states and cell states of LSTM
	def init_hidden(self, batch_size):
		return (Variable(torch.zeros(1, batch_size, self.lstm_size),
						requires_grad=False).type(self.float_type),
				Variable(torch.zeros(1, batch_size, self.lstm_size),
						requires_grad=False).type(self.float_type))
	# Forward process
	def forward(self, rep1, len1, mask1, rep2, len2):
		# Compute context vectors using attention.
		# outputs_1: this is the matrix Y in the paper consisting of the LSMT1 output vectors (may be
		transposed).
		# WyY: this is the product between Y and the matrix Wy in the paper. Use it to compute M as in the
		paper.
		def context_vector(h_t, max_seq_len, outputs_1, WyY):
		WhH = torch.matmul(h_t, self.Wh)
		# Use mask to ignore the outputs of the padding part in premise
		shape = WhH.size()
		WhH = WhH.view(shape[0], 1, shape[1])
		WhH = WhH.expand(shape[0], max_seq_len, shape[1])
		M1 = mask1.type(self.float_type)
		shape = M1.size()
		M = M1.view(shape[0], shape[1], 1).type(self.float_type)
		M = M.expand(shape[0], shape[1], self.lstm_size)
		WhH = WhH * M
		M = torch.tanh(WyY + WhH)
		aW = self.aW.view(1, 1, -1)
		aW = aW.expand(batch_size, max_seq_len, aW.size()[2])
		# Compute batch dot: the first step of a softmax
		batch_dot = M * aW
		batch_dot = torch.sum(batch_dot, 2)
		max_by_column, _ = torch.max(batch_dot, 1)
		max_by_column = max_by_column.view(-1, 1)
		max_by_column = max_by_column.expand(max_by_column.size()[0], max_seq_len)
		batch_dot = torch.exp(batch_dot - max_by_column) * M1
		# Partition function and attention:
		# the second step of a softmax, use mask to ignore the padding
		partition = torch.sum(batch_dot, 1)
		partition = partition.view(-1, 1)
		partition = partition.expand(partition.size()[0], max_seq_len)
		attention = batch_dot / partition
		# compute context vector
		shape = attention.size()
		attention = attention.view(shape[0], shape[1], 1)
		attention = attention.expand(shape[0], shape[1], self.lstm_size)
		cv_t = outputs_1 * attention
		cv_t = torch.sum(cv_t, 1)
		return cv_t

	# Set batch size
	batch_size = rep1.size()[0]
	max_seq_len = rep1.size()[1]
	# Representation of input sentences
	sent1 = self.embedding(rep1)
	sent2 = self.embedding(rep2)
	# Transform sentences representations to:
	# (sequence length * batch size * feqture size)
	sent1 = sent1.transpose(1, 0)
	sent2 = sent2.transpose(1, 0)
	# First LSTM: (sequence lenght * mini batch * lstm size)
	hidden = self.init_hidden(batch_size)
	states = (hidden[0].view(batch_size, -1), hidden[1].view(batch_size, -1))
	lstm_1_outs = []
	cell_states = []
	for i in range(max_seq_len):
		states = self.lstm1(sent1[i], states)
		lstm_1_outs.append(states[0].view(1, batch_size, -1))
		cell_states.append(states[1].view(1, batch_size, -1))
	lstm_1_outs = torch.cat(lstm_1_outs, 0)
	outputs_1 = lstm_1_outs.transpose(0, 1)
	cell_states = torch.cat(cell_states, 0)
	cell_states = cell_states.transpose(1, 0)
	length = (len1-1).view(-1, 1, 1).expand(cell_states.size(0), 1, cell_states.size(2))
	hidden_1 = torch.gather(cell_states, 1, length)
	# Second LSTM: (sequence lenght * mini batch * lstm size)
	# Initialize the cell states of with the last cell states of the first LSTM
	hidden = self.init_hidden(batch_size)
	hidden = (hidden[0], hidden_1.view(1, batch_size, -1))
	lstm_2_outs, hidden = self.lstm2(sent2, hidden)
	outputs_2 = lstm_2_outs.transpose(1, 0)
	length = (len2-1).view(-1, 1, 1).expand(outputs_2.size(0), 1, outputs_2.size(2))
	output = torch.gather(outputs_2, 1, length)
	output = output.view(output.size(0), -1)
	WyY = torch.matmul(outputs_1, self.Wy)
	# compute context vector
	R = context_vector(output, max_seq_len, outputs_1, WyY)
	WpR = torch.matmul(R, self.Wp)
	WxH = torch.matmul(output, self.Wx)
	# compute the final representation of the sentence pair
	h_star = torch.tanh(WpR + WxH)
	h_star = F.dropout(h_star, p=self.drop_out)
	# Fully connected layers
	fc_out = F.dropout(F.tanh(self.linear1(h_star)), p=self.drop_out)
	fc_out = F.dropout(F.tanh(self.linear2(fc_out)), p=self.drop_out)
	fc_out = F.dropout(F.tanh(self.linear3(fc_out)), p=self.drop_out)
	# Fully connected layer for softmax
	fc_out = self.linear4(fc_out)
	return F.log_softmax(fc_out)

	###############################################################
# Recurrent neural network class
class RNNNet(object):
	def __init__(self, mode):
		self.mode = mode
		# Set tensor type when using GPU
		if torch.cuda.is_available():
			self.use_gpu = True
			self.float_type = torch.cuda.FloatTensor
			self.long_type = torch.cuda.LongTensor
		# Set tensor type when using CPU
		else:
			self.use_gpu = False
			self.float_type = torch.FloatTensor
			self.long_type = torch.LongTensor

	# Get a batch of data from given data set.
	def get_batch(self, data_set, s, e):
		sent_1 = data_set[0]
		len_1 = data_set[1]
		sent_2 = data_set[2]
		len_2 = data_set[3]
		label = data_set[4]
		return sent_1[s:e], len_1[s:e], sent_2[s:e], len_2[s:e], label[s:e]

	# Create mask for premise sentences.
	def create_mask(self, data_set, max_length):
		length = data_set[1]
		masks = []
		for one in length:
		mask = list(numpy.ones(one))
		mask.extend(list(numpy.zeros(max_length - one)))
		masks.append(mask)
		masks = numpy.asarray(masks, dtype=numpy.float32)
		return masks
	# Evaluate the trained model on test set
	def evaluate_model(self, pred_Y, Y):
		_, idx = torch.max(pred_Y, dim=1)
		# move tensor from GPU to CPU when using GPU
		if self.use_gpu:
			idx = idx.cpu()
			Y = Y.cpu()
		idx = idx.data.numpy()
		Y = Y.data.numpy()
		accuracy = numpy.sum(idx == Y)
		return accuracy
	# Train and evaluate SNLI models
	def train_and_evaluate(self, FLAGS, embedding, train_set, dev_set, test_set):
		class_num = 3
		num_epochs = FLAGS.num_epochs
		batch_size = FLAGS.batch_size
		learning_rate = FLAGS.learning_rate
		beta = FLAGS.beta
		drop_rate = FLAGS.dropout_rate
		lstm_size = FLAGS.lstm_size
		hidden_size = FLAGS.hidden_size
		# Word embeding
		vectors = embedding.vectors
		# Max length of input sequence
		max_seq_len = train_set[0].shape[1]
		# Create mask for first sentence
		train_mask = self.create_mask(train_set, max_seq_len)
		dev_mask = self.create_mask(dev_set, max_seq_len)
		test_mask = self.create_mask(test_set, max_seq_len)
		# Train, validate and test set size
		train_size = train_set[0].shape[0]
		dev_size = dev_set[0].shape[0]
		test_size = test_set[0].shape[0]
		# Initialize embedding matrix
		embedding = NN.Embedding(vectors.shape[0], vectors.shape[1], padding_idx=0)
		embedding.weight = NN.Parameter(torch.from_numpy(vectors))
		embedding.weight.requires_grad = False
		# uncommet the below three lines to force the code to use CPU
		#self.use_gpu = False
		#self.float_type = torch.FloatTensor
		#self.long_type = torch.LongTensor
		# Define models
		model = eval("Model_" + str(self.mode))(
					self.use_gpu, lstm_size, hidden_size, drop_rate, beta, embedding, class_num
				)
		# If GPU is availabel, then run experiments on GPU
		if self.use_gpu:
			model.cuda()

		# ======================================================================
		# define training operation
		#
		optimizer = OPT.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
							 lr=learning_rate)
		# ======================================================================
		# define accuracy operation
		# ----------------- YOUR CODE HERE ----------------------
		#
		accuracy = 0
		for i in range(num_epochs):
		# put model to training mode
			model.train()

			# Shuffle the train set.
			#idx = list(range(train_size))
			#numpy.random.shuffle(idx)
			#train_set[0] = train_set[0][idx]
			#train_set[1] = train_set[1][idx]
			#train_set[2] = train_set[2][idx]
			#train_set[3] = train_set[3][idx]
			#train_set[4] = train_set[4][idx]
			print(20 * '*', 'epoch', i+1, 20 * '*')
			start_time = time.time()
			s = 0
			while s < train_size:
				model.train()
				e = min(s + batch_size, train_size)
				batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
				self.get_batch(train_set, s, e)
				mask = train_mask[s:e]
				rep1 = Variable(torch.from_numpy(batch_1v),
				requires_grad=False).type(self.long_type)
				len1 = Variable(torch.from_numpy(batch_1l),
				requires_grad=False).type(self.long_type)
				rep2 = Variable(torch.from_numpy(batch_2v),
				requires_grad=False).type(self.long_type)
				len2 = Variable(torch.from_numpy(batch_2l),
				requires_grad=False).type(self.long_type)
				mask = Variable(torch.from_numpy(mask),
				requires_grad=False).type(self.long_type)
				label = Variable(torch.from_numpy(batch_label),
				requires_grad=False).type(self.long_type)
				# Forward pass: predict labels
				pred_label = model(rep1, len1, mask, rep2, len2)
				# Loss function: compute negative log likelyhood
				loss = F.nll_loss(pred_label, label)
				# Zero gradients, perform a backward pass, and update the weights.
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				s = e
			end_time = time.time()
			print ('the training took: %d(s)' % (end_time - start_time))
			# Put model to evaluation mode
			model.eval()
			# Evaluate the trained model on validate set
			s = 0
			total_correct = 0
			while s < dev_size:
				e = min(s + batch_size, dev_size)
				batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
					self.get_batch(dev_set, s, e)
				mask = dev_mask[s:e]

				rep1 = Variable(torch.from_numpy(batch_1v),
								requires_grad=False).type(self.long_type)
				len1 = Variable(torch.from_numpy(batch_1l),
								requires_grad=False).type(self.long_type)
				rep2 = Variable(torch.from_numpy(batch_2v),
								requires_grad=False).type(self.long_type)

				len2 = Variable(torch.from_numpy(batch_2l),
								requires_grad=False).type(self.long_type)
				mask = Variable(torch.from_numpy(mask),
								requires_grad=False).type(self.long_type)
				label = Variable(torch.from_numpy(batch_label),
								requires_grad=False).type(self.long_type)
				# Forward pass: predict labels
				pred_label = model(rep1, len1, mask, rep2, len2)
				total_correct += self.evaluate_model(pred_label, label)
				s = e
			print ('accuracy of the trained model on validation set %f' %
					(total_correct / dev_size))
			print ()
			# evaluate the trained model on test set
			s = 0
			total_correct = 0
			while s < test_size:
				e = min(s + batch_size, test_size)
				batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
					self.get_batch(test_set, s, e)
				mask = test_mask[s:e]
				rep1 = Variable(torch.from_numpy(batch_1v),
								requires_grad=False).type(self.long_type)
				len1 = Variable(torch.from_numpy(batch_1l),
								requires_grad=False).type(self.long_type)
				rep2 = Variable(torch.from_numpy(batch_2v),
								requires_grad=False).type(self.long_type)
				len2 = Variable(torch.from_numpy(batch_2l),
								requires_grad=False).type(self.long_type)
				mask = Variable(torch.from_numpy(mask),
								requires_grad=False).type(self.long_type)
				label = Variable(torch.from_numpy(batch_label),
								requires_grad=False).type(self.long_type)

				# Forward pass: predict labels
				pred_label = model(rep1, len1, mask, rep2, len2)

				total_correct += self.evaluate_model(pred_label, label)

				s = e
		return total_correct / test_size