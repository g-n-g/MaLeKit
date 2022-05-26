# MaLeKit

MaLeKit (Machine Learning Kit) is a library
of some non-standard machine learning algorithms.

-------------------------------------------------------------------------------
# Algorithms

*RidgeRegTree* and *RidgeRegForest*
Ridge Regression Tree and Forest

-------------------------------------------------------------------------------
# Python

The installation of a minimal python virtual environment:

```bash
python -m venv .../pyenv  # creating empty virtual environment
source .../pyenv/bin/activate  # activating the virtual environment

pip install --upgrade pip
pip install --upgrade setuptools

pip install nose
pip install numpy
pip install joblib
```

Then the tests can be run by nose:
```bash
source .../pyenv/bin/activate  # if not done yet
cd .../MaLeKit/python  # go to the python directory of this project
PYTHONPATH=. nosetests --with-doctests
```
