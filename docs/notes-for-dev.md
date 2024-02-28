
# the ml models and their unlearning algorithms

PROJECT STRUCTURE:
```
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
```
# to get started refer this:
- clone the repo properly, try not to change anything, it is suggested that you follow the same directory, file and all those other thing's naming con. as in the repo, follow PEP8 guidelines
- set up a venv
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
- dataset used for sisa part: 
[purchase-dataset](https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz)
[adult](https://archive.ics.uci.edu/dataset/2/adult)
### distribution.py
- type of distribution is boolean -> uniform or non uniform
- making seperate functions for the shards and the requests
### datloader.py and datasetfile should be the part of the framework and ideally one should, let the user decide the parameters for the datasetfie

### notes on splitfile, datasetfile, requestsfile:
- Splitfile: This file contains the indices that divide the dataset into different shards. Shards are subsets of the dataset used for training. Each shard typically represents a portion of the dataset, and the splitfile helps in dividing the dataset into these shards.

- Requestfiles: These files represent the unlearning requests made to the model during training. Unlearning is a process where specific data points or patterns are removed from the model to adapt to changing circumstances or to mitigate biases. Each requestfile corresponds to a shard and contains the indices of the data points that need to be removed from the corresponding shard.

- Datasetfile: This file contains metadata about the dataset, such as the input shape, the number of classes, and the paths to the dataloader scripts. It helps in configuring the training process, such as loading the dataset and defining the model architecture.
### to run sisa:
0. `cd examples/`
1. exec -> `python example_preprocessing.py`
2. `python example_distribution.py` this will create a contaners directory, and a child directory named on the number of shards you chose,
3. `python example_sisa_train.py
` 

### describing your functions:
```
def aggregate_outputs(strategy, container, shards, dataset, label, baseline=None):
    """
    Aggregate outputs from different shards based on the specified strategy.

    Parameters:
        - strategy (str): The voting strategy ('uniform', 'models:', 'proportional').
        - container (str): Path to the container directory.
        - shards (int): Number of shards.
        - dataset (str): Location of the dataset file.
        - label (str): Label for the outputs.
        - baseline (int): Use only the specified shard as a lone shard baseline.

    Returns:
        - np.ndarray: Aggregated votes.
    """
```
