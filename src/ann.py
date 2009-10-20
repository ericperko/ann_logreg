import pdb
import sys
import datetime

import data_parser.parser
import libs.folds
import libs.node
import libs.dtree_predict
import libs.dtree_helper

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
    
    columns2 = columns.copy()
    del columns2[len(columns)-1]
    del columns2[0]

if __name__ == "__main__":

    main(sys.argv[1:])
    columns = {}
    #main(["ab", "1", "10", "0", "0"])
    print "Finished"
