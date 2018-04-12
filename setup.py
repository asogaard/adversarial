from setuptools import setup

setup(
    name = "adversarial",
    version = "0.1",
    author = "Andreas Sogaard",
    author_email = "andreas.sogaard@ed.ac.uk",
    description = ("Tools for training and evaluating de-correlated jet taggers"),
    keywords = "Adversarial neural networks, jet substructure",
    url="https://github.com/asogaard/adversarial",
    packages=['adversarial',
              'optimisation',
              'run',
              'run.adversarial',
              'run.uboost',
              'optimisation',
              ],
    )
