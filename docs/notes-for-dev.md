
# the ml models and their unlearning algorithms

PROJECT STRUCTURE:

repo-root/
|---vershachi/
    |---__init__.py
    |---fisher_unlearning/
    |   |---__init__.py
    |   |---fisher_unlearning.py 
    |---knot_unlearning/
    |   |---__init__.py
    |   |---knot.py
    |---bayesian_unlearning/
    |   |---__init__.py
    |   |---bayesian_unlearning.py
    |---sisa_unlearning/
    |   |---__init__.py
    |   |---sisa.py
    |---utils.py
|---models/
|   |---__init__.py
|   |---model1.py
|   |---model2.py
|---tools/

## to get started refer this:
- to get started you have to build this library in the development mode 
- there is a simple setup.py file which defines our standalone library, in the root of the repo
- run this command in the root of the repo to install the library in the development mode `pip install -e .`

## dataloader.py
- it is being used in the sharded.py file in the module `sisa` in the library
- it is being used so that the before you can train your model, the datasets are loaded perfectly and stuff and such.

## 1. linear regression models:
### refer paper "certifiable mul for linear models"
- learning algos:
  - linear regression

- unlearning algos:
  - fisher unlearning method
  - influence unlearning method
  - 

- datasets that can be used:
  - mnist, cifar, 

## 2. neural network:
- learning algos:
  - gradient descent

- unlearning algos:
  - sisa

## 3. naive bayes:
-  learning algos:

- unlearning algos:

## 4. decision trees:
- learning algos:
  - id3

- unlearning algos:

## 5. federated learning:
-KNOT




## working on sisa
### distribution.py
- type of distribution is boolean -> uniform or non uniform
- making seperate functions for the shards and the requests
