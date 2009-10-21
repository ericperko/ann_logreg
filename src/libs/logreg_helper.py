import math
import random

random.seed(12345)

class LogisticRegression:
    """
    Following the Logistic Regression section in Mitchell's tentative chapter for a new ML book.
    www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf
    """
    def __init__(self, num_inputs, c, learning_rate):
        self.num_inputs = num_inputs
        self.c = c
        self.learning_rate = learning_rate
        self.weights = [random.uniform(-0.1, 0.1) for i in xrange(0, num_inputs+1)]
    
    def train(self, examples):
        last = float("-inf")
        while(True):
            for i in xrange(0, len(self.weights)):
                change = self.calcGradientOfWi(i, examples)
                self.weights[i] = self.weights[i] + self.learning_rate * change
            current = self.logLikelihood(examples)
            if current > last:
                last = current
            else:
                break
                
    def calcGradientOfWi(self, i, examples):
        sum1 = 0
        for example in examples:
            sum1 += self.calcErrorOfLi(i, example)
        sum1 += 2 * self.c * self.weights[i]
        return sum1
        
    def calcErrorOfLi(self, i, example):
        if i == 0:
            input_i = 1
        else:
            input_i = example[i-1]
        error = input_i * (example[-1] - self.probYPos(example))
        return error
    
    def probYPos(self, example):
        sum1 = self.weights[0]
        for i in xrange(0, self.num_inputs):
            sum1 += self.weights[i+1] * example[i]
        numerator = 1
        try:
            denom = 1 + math.exp(-1*sum1)
        except OverflowError:
            denom = 1
        return numerator / denom
    
    def logLikelihood(self, examples):
        sum1 = 0
        for example in examples:
            probYPos = self.probYPos(example)
            probYNeg = 1 - probYPos
            sum1 += example[-1] * logreg_log(probYPos) + (1 - example[-1]) * logreg_log(probYNeg)
        return sum1
    
def logreg_log(x):
    if x == 0:
        return 0
    else:
        return math.log(x)