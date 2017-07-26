import numpy as np
import math

# print np.dot(w0, w1)
sigmoid = lambda x : 1 / (1 + np.exp(-x))

class MLP:

	def __init__(self, num_Input, num_Hidden, num_Output, learn_Rate, num_Epoch):
		self.w0 = np.random.random((num_Input + 1, num_Hidden))
		self.w1 = np.random.random((num_Hidden + 1, num_Output))
		self.learn_Rate = learn_Rate
		self.num_Epoch = num_Epoch

	def feedforward(self, input_Vector):
		input_Vector = [input_Vector]
		input_Vector = np.hstack(([[-1]], input_Vector))

		i1 = np.dot(input_Vector, self.w0)
		y1 = sigmoid(i1)
		y1 = np.hstack(([[-1]], y1))
		

		i2 = np.dot(y1, self.w1)
		y2 = sigmoid(i2)
		# print y2
		return y2

	def backpropagation(self, train_Set, label):

		for j in xrange(self.num_Epoch):
			err = 0

			for i in range(len(label)):

				# feedforward
				input_Vector = np.hstack(([[-1]], [train_Set[i]]))
				i1 = np.dot(input_Vector, self.w0)
				y1 = sigmoid(i1)
				y1 = np.hstack(([[-1]], y1))
				

				i2 = np.dot(y1, self.w1)
				y2 = sigmoid(i2)

				# backpropagation

				delta2 = (label[i] - y2) * (y2 * (1 - y2))
				deltaW1 = delta2 * self.learn_Rate * np.transpose(y1)

				self.w1 = self.w1 + deltaW1
			
				delta1 = (delta2 * self.w1[1:]) * (np.transpose(y1)[1:] * (1 - np.transpose(y1)[1:]))
				dw1 = delta1 * self.learn_Rate * input_Vector

				self.w0 = self.w0 + np.transpose(dw1)
				
				err += math.fabs(label[i] - y2[0][0])
		

mlp = MLP(2, 2, 1, 0.7, 3000)

# mlp.forward([[0.3, 0.3]])

train_Set = [[0, 0],
			 [0, 1],
			 [1, 0],
			 [1, 1]]

label = [0, 1, 1, 0]

print "Before\n"
for i in range(len(train_Set)):
	print (mlp.feedforward(train_Set[i])[0][0], label[i])

mlp.backpropagation(train_Set, label)

print "After\n"
for i in range(len(train_Set)):
	print (mlp.feedforward(train_Set[i])[0][0], label[i])