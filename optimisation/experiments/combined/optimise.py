# Basic import(s)
import os
import numpy as np

# Project import(s)
import adversarial
from adversarial.utils import parse_args
from optimisation.common import *
from run.adversarial import train

# Global variables
PROJECTDIR='/'.join(os.path.realpath(adversarial.__file__).split('/')[:-2] + [''])

# Main function, called by the Spearmint optimisation procedure
def main(job_id, params):

    # Logging
    print "Call to main function (#{})".format(job_id)
    print "  Parameters: {}".format(params)

    # Create temporary patch dictionary
    jobname  = 'patch.{:08d}'.format(job_id)
    filename = os.path.realpath('patches/{}.json'.format(jobname))
    patch    = create_patch(params)

    # Adversarial-specific change
    lr_ratio = patch['combined']['model'].pop('lr_ratio')
    if lr_ratio < 0:
        lr_ratio = np.power(10., lr_ratio)
        pass
    patch['combined']['compile']['loss_weights'] = [lr_ratio, 1.0]

    # -- Fixed settings
    for field in ['fit', 'model']:
        if field not in patch['combined']:
            patch['combined'][field] = dict()
            pass
        pass
    patch['combined']['pretrain']            = 20
    patch['combined']['fit']['epochs']       = 200
    patch['combined']['fit']['batch_size']   = 8192
    patch['combined']['model']['lambda_reg'] = 10.

    # Save patch to file
    save_patch(patch, filename)

    # Set arguments
    args = parse_args(['--optimise-adversarial',
                       '--patch',   filename,
                       '--jobname', 'combined-' + jobname,
                       '--gpu',
                       '--devices', '7',
                       '--folds',   '3',
                       '--tensorboard'],
                      adversarial=True)

    # Call main script (in the correct directory)
    with cd(PROJECTDIR):
        result = train.main(args)
        pass

    # Ensure correct type, otherwise Spearmint does not accept value
    result = float(result)
    
    return result
