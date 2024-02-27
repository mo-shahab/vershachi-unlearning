from importlib import import_module
from vershachi.sisa.sisa import SisaTrainer

# Define the path to the model directory
model_dir = r"C:\dev\vershachi-unlearning\models"

# Define the path to the dataset file
dataset_file = r"C:\dev\vershachi-unlearning\datasets\datasetfile"

# Number of shards
shards = 3

# Create an instance of SisaTrainer for testing
tester = SisaTrainer(model_dir=model_dir, dataset_file=dataset_file, test=True)

# Test each shard
for shard in range(shards):
    for j in range(5):
        print(f"Testing shard: {shard+1}/{shards}, iteration: {j+1}/5")
        # Determine the label for this test iteration
        r = j * shards // 5
        # Create a new instance of SisaTrainer for testing this shard and iteration
        tester = SisaTrainer(
            model_dir=model_dir,
            dataset_file=dataset_file,
            container=shards,
            shard=shard,
            label=str(r),
            test=True,
        )
        # Perform testing
        tester._test()

