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

	def feedforward(self, input_Vector):
		input_Vector = [input_Vector]
		input_Vector = np.hstack(([[-1]], input_Vector))

		i1 = np.dot(input_Vector, self.w0)
		y1 = sigmoid(i1)
		y1 = np.hstack(([[-1]], y1))
		

		i2 = np.dot(y1, self.w1)
		y2 = sigmoid(i2)
		
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

		prev_Step2 = 0
		prev_Step1 = 0

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

				delta2 = (label[i] - y2) * (y2 * (1 - y2))
				curr_Delta2 += delta2 
				curr_Grad2 += delta2 * np.transpose(y1)


				delta1 = (delta2 * self.w1[1:]) * (np.transpose(y1)[1:] * (1 - np.transpose(y1)[1:])) 
				
				curr_Delta1 += delta1 
				curr_Grad1 += delta1 * input_Vector

			if (type(prev_Grad2) == int):

				prev_Step2 = curr_Grad2 * self.learn_Rate
				prev_Step1 = np.transpose(curr_Grad1) * self.learn_Rate

				self.w1 = self.w1 + curr_Grad2
				self.w0 = self.w0 + np.transpose(curr_Grad1)
			else:
				#  for (i=0; i<length; i++) {
				#   momentum[i] = gradient[i]/(lastGradient[i] - gradient[i]);
				#   if (momentum[i] > defaultMomentum)
				#    momentum[i] = defaultMomentum;
				#   if (momentum[i] < -defaultMomentum)
				#    momentum[i] = -defaultMomentum;
				#   delta[i] = momentum[i]*delta[i] + lastGradient[i]*(gradient[i]<0?0:step[individual?i:0]*gradient[i]);
				#   weights[i] += delta[i];
				#   lastGradient[i] = gradient[i];
				#  }
				# }
				# print "curgrad2"
				# print curr_Grad2
				# print "curdelta2"
				# print curr_Delta2
				# print 
				# http://venturas.org/sites/venturas.org/files/mydmbp.pdf
				shrink = self.max_Growth_Factor / (1 + self.max_Growth_Factor)
				new_W1 = np.zeros((len(curr_Grad2), len(curr_Grad2[0])))
				for i in range(len(curr_Grad2)):
					for j in range(len(curr_Grad2[0])):
						if curr_Grad2[i][j] > 0:
							if curr_Delta2[i][j] > shrink * prev_Delta2[i][j]:
								new_W1[i][j] = self.max_Growth_Factor * curr_Grad2[i][j]

							elif curr_Delta2[i][j] < shrink * prev_Delta2[i][j]:
								new_W1 = (curr_Delta2[i][j] / 
									(prev_Delta2[i][j] - curr_Delta2[i][j])) * curr_Grad2[i][j]

							# elif 


				new_W1 = curr_Grad2 + (curr_Grad2 / (prev_Grad2 - curr_Grad2)) * prev_Step2
				
				# smaller_Same_Sign2 = 


				boolean1 = new_W1 > prev_Step2 * self.max_Growth_Factor
				new_W1[boolean1] = self.w1[boolean1] * self.max_Growth_Factor

				boolean1 = new_W1 < prev_Step2 * -self.max_Growth_Factor
				new_W1[boolean1] = self.w1[boolean1] * -self.max_Growth_Factor

				prev_Step2 = new_W1
				self.w1 += new_W1
				



				new_W0 = np.transpose(curr_Grad1) + np.transpose(curr_Grad1 / (prev_Grad1 - curr_Grad1)) * prev_Step1
				
				
				boolean0 = new_W0 > prev_Step1 * self.max_Growth_Factor
				new_W0[boolean0] = self.w0[boolean0] * self.max_Growth_Factor

				boolean0 = new_W0 < prev_Step1 * -self.max_Growth_Factor
				new_W0[boolean0] = self.w0[boolean0] * -self.max_Growth_Factor


				prev_Step1 = new_W0
				self.w0 += new_W0


			prev_Delta2 = curr_Delta2
			prev_Grad2 = curr_Grad2

			prev_Grad1 = curr_Grad1
			prev_Delta1 = curr_Delta1

mlp = MLP(2, 2, 1, 0.7, 1000)

train_Set = [[1, 1],
			 [0, 1],
			 [1, 0],
			 [1, 1]]

label = [0, 1, 1, 0]

# print mlp.feedforward(train_Set[0])

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

# print "W0: \n"
# print mlp.w0
# print "\nW1 \n"
# print mlp.w1

# train = open("data/training.txt")
# data = train.readlines()
# train_Set = [map(lambda x : int(x), line.split(",")[1:]) for line in data]
# label = [int(line.split(",")[0]) for line in data]

# mlp = MLP(len(train_Set[0]), 10, 26, 0.7, 1000)

# print mlp.feedforward(train_Set[0])

# mlp.backpropagation(train_Set, label)