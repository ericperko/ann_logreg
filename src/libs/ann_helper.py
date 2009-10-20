import math
import random

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate, weight_decay_gamma):
        self.weights, self.inputs = build_weights_and_inputs_matrices(num_inputs, num_hidden, num_outputs)
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.weight_decay_gamma = weight_decay_gamma
    
    def feedforward(self, example):
        outputs = {}
        for j in range(0, self.num_hidden):
            for i in range(0, self.num_inputs):
                self.inputs[j][i] = example[i]
        for j in range(self.num_hidden, self.num_hidden+self.num_outputs):
            sum2 = 0
            for i in range(0, self.num_hidden):
                sum1 = 0
                for k in range(0, self.num_inputs):
                    sum1 += self.inputs[i][k] * self.weights[i][k]
                self.inputs[j][i] = sigmoid(sum1)
                sum2 += self.inputs[j][i] * self.weights[j][i]
            outputs[j] = sigmoid(sum2)
        return outputs
    
    def backprop(self, examples, num_iterations):
        for q in range(0, num_iterations):
            total_error = 0
            for example in examples:
                errors = {}
                outputs= self.feedforward(example)
                for k in range(self.num_hidden, self.num_hidden + self.num_outputs):
                    errors[k] = outputs[k] * (1 - outputs[k]) * (example[-1] - outputs[k])
                    total_error += errors[k]
                for h in range(0, self.num_hidden):
                    output_h = self.inputs[-1][h]
                    sum1 = 0
                    for k in range(self.num_hidden, self.num_hidden + self.num_outputs):
                        sum1 += self.weights[k][h] * errors[k]
                    errors[h] = output_h * (1 - output_h) * sum1
                for j in range(self.num_hidden, self.num_hidden + self.num_outputs):
                    for i in range(0, self.num_hidden):
                        w_t1 = (1 - 2*self.learning_rate*self.weight_decay_gamma) * self.weights[j][i] #weight decay
                        w_t2 = self.learning_rate * errors[j] * self.inputs[j][i]
                        self.weights[j][i] = w_t1 + w_t2
                for j in range(0, self.num_hidden):
                    for i in range(0, self.num_inputs):
                        w_t1 = (1 - 2*self.learning_rate*self.weight_decay_gamma) * self.weights[j][i] #weight decay
                        w_t2 = self.learning_rate * errors[j] * self.inputs[j][i]
                        self.weights[j][i] = w_t1 + w_t2
            total_error = total_error / len(examples)
            if math.fabs(total_error) < 0.000000001: # really small total error...
                break
                        
def sigmoid(x):
    """
    Sigmoid activation function. 
    """
    val = 1.0 / (1.0 + math.exp(-1*x))
    return val
    
def build_weights_and_inputs_matrices(num_inputs, num_hidden, num_outputs):
    random.seed(12345)
    weights = []
    inputs = []
    for i in range(0, num_hidden):
        weights.append([random.uniform(-0.1, 0.1) for i in range(0, num_inputs)])
        inputs.append([0 for i in range(0, num_inputs)])
    for i in range(num_hidden, num_hidden + num_outputs):
        weights.append([random.uniform(-0.1, 0.1) for i in range(0, num_hidden)])
        inputs.append([0 for i in range(0, num_hidden)])
    return (weights, inputs)

if __name__ == "__main__":
    n = NeuralNetwork(10, 25, 1, 0.1, 0.1)
    sample = [1 for i in range(0, 11)]
    n.backprop([sample], 1000)