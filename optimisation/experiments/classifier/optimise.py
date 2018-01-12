# Basic import(s)
import os
import sys
import json

# Project import(s)
sys.path.append(os.path.abspath('../../../'))  # This is pretty bad practice...
import run


class cd:
    """Context manager for changing the current working directory.
    From: [https://stackoverflow.com/a/13197763]"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
        return

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
        return

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        return
    pass


def create_patch (params, path):
    """Create temporary patch file to use in optimisation.

    The method traverses the entries in dict `params`, which have a hierarchial
    structure
        params = { ..., 'first-level/second-level/third-level': value, ... }

    The patch is saved as JSON file in `path` with structure
        # JSON file `path`
        ...
        "first-level" : {
            "second-level" : {
                "third-level" : value,
                ...
            }
            ...
        }
        ...
    """

    # Initialise output dictionary
    result = dict()

    # Loop parameters, estable dictionary structure
    for param in params:
        print "param: {}".format(param)
        entry = result

        # Traverse hierarchy
        keys = param.split('/')
        for key in keys[:-1]:
            if key not in entry:
                entry[key] = dict()
                pass
            entry = entry[key]
            pass

        # Set deepest-level entry by assignment
        entry[keys[-1]] = params[param][0]
        pass

    # Save to JSON
    print "- " * 40
    print "Saving the folloing patch to '{}':".format(path)
    print result
    print "- " * 40

    # -- Make sure target directory exists
    if '/' in path:
        directory = '/'.join(path.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
            pass
        pass

    # -- Dump JSON to file
    with open(path, 'w') as patchfile:
        json.dump(result, patchfile, indent=4, sort_keys=True)
        pass

    return


# Main function, called by the Spearmint optimisation procedure
def main(job_id, params):

    # Logging
    print "Call to main function (#{})".format(job_id)
    print "  Parameters: {}".format(params)

    # Create temporary patch file
    patch = os.path.realpath('patches/patch.{:08d}.json'.format(job_id))
    create_patch(params, patch)

    # Set arguments
    # @TODO: Dynamically decide `--gpu`, `--devices N`?
    args = run.parse_args(['--optimise-classifier', '--patch', patch, '--tensorflow', '--gpu', '--devices', '3', '--folds', '5'])
    print args

    # Call main script from the correct directory
    run_dir = os.path.realpath('/'.join(run.__file__.split('/')[:-1]))
    print "Calling `run.py` in {}.".format(run_dir)
    with cd(run_dir):
        result = run.main(args)
        pass

    # Ensure correct type, otherwise Spearmint does not accept value
    result = float(result)
    return result

