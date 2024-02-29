"""
To define your custom model use this script,
import necessary modules and libraries before you 
tailor any module.

Not the complete script, a skeleton to what all will 
be needed and what all should be done to create your 
own Neural Network model.
"""


# Define CustomModel class
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define your custom model architecture here

    def forward(self, x):
        # Define the forward pass of your model here
        return x


# Update model_init function to initialize CustomModel
def model_init(data_name):
    if data_name == "mnist":
        model = Net_mnist()
    elif data_name == "cifar10":
        model = Net_cifar10()
    elif data_name == "purchase":
        model = Net_purchase()
    elif data_name == "adult":
        model = Net_adult()
    elif data_name == "custom_dataset":
        model = CustomModel()  # Initialize CustomModel for custom dataset
    else:
        raise ValueError("Unknown dataset name: {}".format(data_name))
    return model
