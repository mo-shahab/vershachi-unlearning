# Import necessary modules
from importlib import import_module
from vershachi.sisa.sisa import SisaTrainer


# the path to the model directory
# Define the path to the model directory
# model_dir = '../models/'
model_dir = r'C:\dev\vershachi-unlearning\models'
# Define the path to the dataset file
dataset_file = r'C:\dev\vershachi-unlearning\datasets\datasetfile'
shards = 4 # should avoid hardcoding the number of shards and such like this

# Create an instance of Sisa_Trainer and train
trainer = SisaTrainer(model_dir=model_dir, dataset_file=dataset_file, train=True)
for i in range(shards):
    for j in range(16):
        print(f"shard: {i+1}/{shards}, requests: {j+1}/16")
        r = j * shards // 5
        trainer = SisaTrainer(model_dir=model_dir, dataset_file=dataset_file, shard=i, label=str(r))
        trainer._train()
