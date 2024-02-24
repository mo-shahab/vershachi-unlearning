import json
from sisa import SISA

# Define parameters
model = "purchase"
train = True
test = False
epochs = 20
batch_size = 16
learning_rate = 0.001
optimizer = "sgd"
output_type = "argmax"
container = "your_container_name"
shard = 1
slices = 1
dataset = "path_to_your_dataset_file"
label = "latest"

# Create SISA object
sisa = SISA(
    model,
    dataset,
    container,
    shard,
    slices,
    label=label,
    train=train,
    test=test,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    optimizer=optimizer,
    output_type=output_type,
)

# Train model
if train:
    sisa.train()

# Test model
if test:
    sisa.test()
