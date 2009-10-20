from __future__ import division
import statlib.stats
import pdb

def test_ann(net, examples):
    result = Result()
    for example in examples:
        output = net.feedforward(example)
        label = output.values()[0]
        result.addLabeled(example, label)
    return result

def calc_roc_area(results):
    roc_vals = {}
    examples = []
    for result in results:
        examples.extend(result.examples)
    total_pos = len(filter(lambda(x): x[0] == 1, examples))
    total_neg = len(filter(lambda(x): x[0] == 0, examples))
    total = len(examples)
    sum1 = 0;
    last_p = (0, 0)
    i = 0.9
    while(i > 0):
        pos_candidates = filter(lambda(x): x[1] >= i, examples)
        tp = len(filter(lambda(x): x[0] == 1, pos_candidates))
        fp = len(pos_candidates) - tp
        tpr = tp/total_pos
        fpr = fp/total_neg
        area = (tpr - last_p[0]) * (fpr - last_p[1])
        last_p = (tpr, fpr)
        sum1 += area
        i -= 0.1
    return sum1
        

def aggregate_results(results):
    accuracies = map(lambda(x): x.accuracy(), results)
    weight_accuracies = map(lambda(x): x.weightedAccuracy(), results)
    precisions = map(lambda(x): x.precision(), results)
    recalls = map(lambda(x): x.recall(), results)

    avg_accuracy = statlib.stats.lmean(accuracies)
    avg_wacc = statlib.stats.lmean(weight_accuracies)
    avg_prec = statlib.stats.lmean(precisions)
    avg_rec = statlib.stats.lmean(recalls)
    
    print "Accuracy: {0:0.3} {1:0.3}\n".format(avg_accuracy, statlib.stats.lstdev(accuracies))
    print "Weighted Accuracy: {0:0.3} {1:0.3}\n".format(avg_wacc, statlib.stats.lstdev(weight_accuracies))
    print "Precision: {0:0.3} {1:0.3}\n".format(avg_prec, statlib.stats.lstdev(precisions))
    print "Recall: {0:0.3} {1:0.3}\n".format(avg_rec, statlib.stats.lstdev(recalls))
    print "Area under ROC: {0:0.3}\n".format(calc_roc_area(results))
    

class Result:
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.examples = []

    def addLabeled(self, example, label):
        self.examples.append((example[-1], label))
        if example[-1] == 1:
            if label >= 0.5:
                self.true_pos += 1
            else:
                self.false_neg += 1
        else:
            if label < 0.5:
                self.true_neg += 1
            else:
                self.false_pos += 1

    def accuracy(self):
        num = (self.true_neg + self.true_pos)
        denom = (self.true_neg + self.true_pos + self.false_neg + self.false_pos)
        return num/denom

    def weightedAccuracy(self):
        term1 = self.true_pos/(self.true_pos + self.false_neg)
        term2 = self.true_neg/(self.true_neg + self.false_pos)
        return ((term1 + term2)/2)

    def precision(self):
        try:
            retval = self.true_pos/(self.true_pos + self.false_pos)
        except ZeroDivisionError:
            retval = 0
        return retval

    def recall(self):
        return self.true_pos/(self.true_pos + self.false_neg)
    
        
        
