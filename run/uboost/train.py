#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training uBoost classifier for de-correlated jet tagging."""

# Basic import(s)
import pickle

# Scientific import(s)
from hep_ml.uboost import uBoostBDT, uBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Project import(s)
from adversarial.new_utils import parse_args, initialise, load_data
from adversarial.profile import profile, Profile


# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data(args.input + 'data.h5')

    # Subset
    data = data.head(1000000)

    # Config
    cfg = {
        'DecisionTreeClassifier': {
            'max_depth': 4,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'uBoostBDT': {  # uBoostClassifier
            'n_estimators': 500,
            'n_neighbors': 50,

            #'efficiency_steps': 3,

            'smoothing': 0.0,
            'uniforming_rate': 1.,
            'learning_rate': 1.
        }
    }


    # Fit uBoost classifier
    # --------------------------------------------------------------------------
    with Profile("Fitting uBoost classifier"):

        # Arrays
        X_train = data.loc[ data['train'],:]
        X_test  = data.loc[~data['train'],:]
        w_train = np.array(data.loc[ data['train'], ['weight']]).flatten()
        w_test  = np.array(data.loc[~data['train'], ['weight']]).flatten()
        y_train = np.array(data.loc[ data['train'], ['signal']]).flatten()
        y_test  = np.array(data.loc[~data['train'], ['signal']]).flatten()
        d_train = data.loc[ data['train'], ['m']]
        d_test  = data.loc[~data['train'], ['m']]

        # Create uBoost classifier
        base_tree = DecisionTreeClassifier(random_state=21,  # For reproducibility
                                           **cfg['DecisionTreeClassifier'])

        clf = uBoostBDT(base_estimator=base_tree,
                        uniform_label=0,
                        uniform_features=['m'],
                        train_features=features,
                        random_state=21,  # For reproducibility
                        target_efficiency=0.6,
                        #n_threads=16,
                        **cfg['uBoostBDT'])

        # Fitr uBoost classifier
        clf.fit(X_train, y_train, sample_weight=w_train)
        pass


    # Saving uBoost classifier
    # --------------------------------------------------------------------------
    print "NOT SAVING ANYTHING"
    #### @TEMP
    #### with Profile("Saving uBoost classifier"):
    ####     with open('models/uboost/uboost.pkl', 'w') as f:
    ####         pickle.dump(clf, f)
    ####         pass
    ####     pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
