# Import necessary modules
from importlib import import_module
from vershachi.sisa.sisa import SisaTrainer


# the path to the model directory
model_dir = '../models/'
# Define a class to simulate command-line arguments
class Args:
    def __init__(self):
        self.model = "purchase"
        self.train = True
        self.test = False
        self.epochs = 20
        self.batch_size = 16
        self.dropout_rate = 0.4
        self.learning_rate = 0.001
        self.optimizer = "sgd"
        self.output_type = "argmax"
        self.container = "conatainers"
        self.shard = 4  # Specify the shard index
        self.slices = 1
        self.dataset = "datasets/purchase/datasetfile"
        self.chkpt_interval = 1
        self.label = "latest"

# Create an instance of Args class
args = Args()

# Create an instance of SISA_Trainer and train
trainer = SisaTrainer(args, model_dir)
trainer.train()
