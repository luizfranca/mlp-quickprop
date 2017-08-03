import numpy as np
import math

# print np.dot(w0, w1)
sigmoid = lambda x : 1 / (1 + np.exp(-x))

class MLP:

	def __init__(self, num_Input, num_Hidden, num_Output, learn_Rate, num_Epoch, max_Growth_Factor = 1.75):
		self.w0 = np.random.random((num_Input + 1, num_Hidden))
		self.w1 = np.random.random((num_Hidden + 1, num_Output))
		self.learn_Rate = learn_Rate
		self.max_Growth_Factor = max_Growth_Factor
		self.num_Epoch = num_Epoch
		self.boolean1 = 0
		self.boolean0 = 0

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

	def quickprop(self, train_Set, label):
		prev_Grad2 = 0
		prev_Grad1 = 0

		prev_Delta2 = 0
		prev_Delta1 = 0



		for j in xrange(self.num_Epoch):
			err = 0

			curr_Delta2 = 0
			curr_Delta1 = 0

			curr_Grad2 = 0
			curr_Grad1 = 0

			for i in range(len(label)):

				# feedforward
				input_Vector = np.hstack(([[-1]], [train_Set[i]]))
				i1 = np.dot(input_Vector, self.w0)
				y1 = sigmoid(i1)
				y1 = np.hstack(([[-1]], y1))
				

				i2 = np.dot(y1, self.w1)
				y2 = sigmoid(i2)

				# quickprop

				# delta2 = (label[i] - y2) * (y2 * (1 - y2))
				# deltaW1 = delta2 * self.learn_Rate * np.transpose(y1)

				# self.w1 = self.w1 + deltaW1
			
				# delta1 = (delta2 * self.w1[1:]) * (np.transpose(y1)[1:] * (1 - np.transpose(y1)[1:]))
				# dw1 = delta1 * self.learn_Rate * input_Vector

				# self.w0 = self.w0 + np.transpose(dw1)



				# new_Delta2 = (label[i] - y2) * (y2 * (1 - y2))
				# new_Grad2 = new_Delta2 * np.transpose(y1)
				delta2 = (label[i] - y2) * (y2 * (1 - y2))
				curr_Delta2 += delta2
				curr_Grad2 += delta2 * self.learn_Rate * np.transpose(y1)

				# new_Delta1 = (new_Delta2 * self.w1[1:]) * (np.transpose(y1)[1:] * (1 - np.transpose(y1)[1:]))
				# new_Grad1 = new_Delta1 * input_Vector

			
				delta1 = (delta2 * self.w1[1:]) * (np.transpose(y1)[1:] * (1 - np.transpose(y1)[1:]))
				curr_Delta1 += delta1
				curr_Grad1 += delta1 * self.learn_Rate * input_Vector

			# if (delta2 )
			## d
			if (type(prev_Grad2) == int):

				self.w1 = self.w1 + curr_Grad2
				self.w0 = self.w0 + np.transpose(curr_Grad1)
				# print "="*30 + " self.w0 first"
				# print self.w0
				# print "="*30 + " w1 thingy"
			else:
				new_W1 = (curr_Delta2 / (prev_Delta2 - curr_Delta2)) * prev_Grad2
				# print curr_Delta2
				# print prev_Delta2

				# print (curr_Delta2 / (prev_Delta2 - curr_Delta2))
				self.boolean1 = new_W1 > self.w1 * self.max_Growth_Factor
				new_W1[self.boolean1] = self.w1[self.boolean1] * self.max_Growth_Factor

				self.boolean1 = new_W1 < self.w1 * -self.max_Growth_Factor
				new_W1[self.boolean1] = self.w1[self.boolean1] * -self.max_Growth_Factor

				# print "new_W1"
				# print new_W1
				self.w1 = new_W1
				# print "="*30 + " transpose first part"
				# print np.transpose(curr_Delta1 / (prev_Delta1 - curr_Delta1))
				# print "="*30  + " prev_Grad1"

				# print prev_Grad1
				# print "="*30  + " self.w0 second"
				# print self.w0
				# print "="*30
				new_W0 = np.transpose(curr_Delta1 / (prev_Delta1 - curr_Delta1)) * np.transpose(prev_Grad1)
				# print "before"
				# print new_W0
				# print ""
				# print new_W0
				# print "\n\n"
				
				
				
				self.boolean0 = new_W0 > self.w0 * self.max_Growth_Factor
				
				new_W0[self.boolean0] = self.w0[self.boolean0] * self.max_Growth_Factor

				self.boolean0 = new_W0 < self.w0 * -self.max_Growth_Factor
				
				new_W0[self.boolean0] = self.w0[self.boolean0] * -self.max_Growth_Factor

				self.w0 = new_W0

				# print "\n\n"
				# print "boolean0"
				# print self.boolean0
				# print "boolean1"
				# print self.boolean1
				# print "new_W0"
				# print new_W0
				# print "new_W1"
				# print new_W1
				# print "self.w0"
				# print self.w0
				# print "self.w1"
				# print self.w1


			prev_Delta2 = curr_Delta2
			prev_Grad2 = curr_Grad2

			prev_Grad1 = curr_Grad1
			prev_Delta1 = curr_Delta1

mlp = MLP(2, 2, 1, 0.7, 1000)

train_Set = [[0, 0],
			 [0, 1],
			 [1, 0],
			 [1, 1]]

label = [0, 1, 1, 0]

# print "Before\n"
# for i in range(len(train_Set)):
# 	print (mlp.feedforward(train_Set[i])[0][0], label[i])

# print mlp.w0
# print mlp.w1

# mlp.backpropagation(train_Set, label)
mlp.quickprop(train_Set, label)

print "After\n"
for i in range(len(train_Set)):
	print (mlp.feedforward(train_Set[i])[0][0], label[i])

print "W0: \n"
print mlp.w0
print "\nW1 \n"
print mlp.w1