import pdb
import sys
import datetime

import data_parser.parser
import libs.folds
import libs.node
import libs.logreg_predict
import libs.logreg_helper

def main(args):
    problem_name, option1, option4 = args
    if not problem_name:
        print "You must specify a problem name."
    try:
        c = float(option1)
    except ValueError:
        print "You must specify an integer for option 1. You specified {0}".format(option1)
    if int(option4) >= 1:
        num_folds = int(option4)
    else:
        print "Option 2 must be a nonnegative integer. You specified {0}".format(option4)

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
        logreg = libs.logreg_helper.LogisticRegression(num_inputs, c, learning_rate)
        logreg.train(examples)
        result = libs.logreg_predict.test_logreg(logreg, stratified_folds[i].examples())
        results.append(result)
    
    libs.logreg_predict.aggregate_results(results)

if __name__ == "__main__":
    main(sys.argv[1:])
    columns = {}
    print "Finished"
