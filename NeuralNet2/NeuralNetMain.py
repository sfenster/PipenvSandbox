'''
Created on Mar 25, 2017

@author: ScottFenstermaker
'''
import numpy
#scipi.special for the sigmoid function expit()
import scipy.special
import matplotlib.pyplot

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# neural network class defintion
class neuralNetwork:
    
    #initialize the neural Network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who (input-to-hidden, hidden-to-output)
        #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        #learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    #train the neural network 
    def train(self, inputs_list, targets_list):
        # convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #calculate signals into the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        
        #calculate the signals immerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        #calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # error os the (target - actual)
        output_errors = targets - final_outputs
        
        # hidden layer error is the output_errors, split by weights, recombined at the hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        #update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    
    #query the neural network 
    def query(self, inputs_list):
        # convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T 
        
        #calculate signals into the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        
        #calculate the signals immerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        #calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        
        pass
    
# network testing parameters
#input_nodes = 3
#hidden_nodes = 3
#output_nodes = 3
#
#learning_rate = 0.3
#
#n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#
#print (n.query([1.0, 0.5, -1.5]))

#open the truncated training file for the MNIST database
data_file = open("mnist_train_10.csv", 'r')
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')