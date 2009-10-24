import pdb
import sys
import datetime

import data_parser.parser
import libs.folds
import libs.ann_helper
import libs.ann_predict

def main(args):
    problem_name, option1, option2, option3, option4 = args
    if not problem_name:
        print "You must specify a problem name."
    try:
        num_hidden_units = int(option1)
    except ValueError:
        print "You must specify an integer for option 1. You specified {0}".format(option1)
    try:
        num_iterations = int(option2)
    except ValueError:
        print "Option 2 must be an integer. You specified {0}".format(option2)
    try:
        weight_decay_gamma = float(option3)
    except ValueError:
        print "Option 3 must be a float. You specified {0}".format(option3)
    if int(option4) >= 1:
        num_folds = int(option4)
    else:
        print "Option 4 must be a nonnegative integer. You specified {0}".format(option4)

    learning_rate = 0.1
        
    columns, data = data_parser.parser.parse_to_logn_and_normalize(problem_name)
    stratified_folds = libs.folds.stratify_folds(num_folds, data)
    
    results = []
    
    for i in range(0, num_folds):
        print "Starting on fold {0}\n".format(i)
        examples = []
        for j in range(0, num_folds):
            if i != j:
                examples.extend(stratified_folds[j].examples())
        num_inputs = len(examples[0]) -1
        n = libs.ann_helper.NeuralNetwork(num_inputs, num_hidden_units, 1, learning_rate, weight_decay_gamma)
        if num_iterations not in range(10000, 100001):
            if num_iterations < 10000:
                num_iterations = 10000
            else:
                num_iterations = 100000
        n.backprop(examples, num_iterations)
        result = libs.ann_predict.test_ann(n, stratified_folds[i].examples())
        results.append(result)
    
    libs.ann_predict.aggregate_results(results)

if __name__ == "__main__":
    main(sys.argv[1:])
    columns = {}
    print "Finished"
