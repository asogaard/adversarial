#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for assessing feature importance."""

# Scientific import(s)
import xgboost as xgb

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data
from adversarial.profile import profile


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, _, _ = load_data(args.input + 'data.h5')

    msk = data['train'] == 1

    features = filter(lambda s: s.startswith('fjet_'), list(data))

    X = data[features]
    y = data['signal']
    w = data['mcEventWeight']

    dtrain = xgb.DMatrix(X[ msk], label=y[ msk], weight=w[ msk])
    dtest  = xgb.DMatrix(X[~msk], label=y[~msk], weight=w[~msk])

    param = {'max_depth':4, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    num_round = 100
    bst = xgb.train(param, dtrain, num_round)

    # make prediction
    preds = bst.predict(dtest)
    importance = bst.get_fscore()

    for name, score in sorted(list(importance.iteritems()), key=lambda t:t[1], reverse=True):
        print "  {:15s}: {:4.1f}".format(name, score)
        pass
    
    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
