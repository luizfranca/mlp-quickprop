import numpy as np
import math

# print np.dot(w0, w1)
sigmoid = np.vectorize(lambda x : 1 / (1 + math.exp(-x)))

class MLP:

	def __init__(self, num_Input, num_Hidden, num_Output):
		self.w0 = np.random.random((num_Input + 1, num_Hidden))
		self.w1 = np.random.random((num_Hidden + 1, num_Output))

	def forward(self, input_Vector):
		input_Vector = np.hstack(([[-1]], input_Vector))

		i1 = np.dot(input_Vector, self.w0)
		y1 = sigmoid(i1)
		y1 = np.hstack(([[-1]], y1))
		

		i2 = np.dot(y1, self.w1)
		y2 = sigmoid(i2)
		print y2
		

mlp = MLP(2, 2, 1)

mlp.forward([[0.3, 0.3]])
