import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,TensorDataset
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

"""Function: load data"""
def data_init(FL_params):

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)
    # shadow_split_idx = [int(whole_trainset.__len__()/2), int(whole_trainset.__len__()) -int(whole_trainset.__len__()/2)]
    # trainset, shadow_trainset = torch.utils.data.random_split(whole_trainset, shadow_split_idx)

    # shadow_split_idx = [int(whole_testset.__len__()/2), int(whole_testset.__len__()) -int(whole_testset.__len__()/2)]
    # testset, shadow_testset = torch.utils.data.random_split(whole_testset, shadow_split_idx)
    #

    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)
    # shadow_test_loader = DataLoader(shadow_testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)


    split_index = [int(trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)

    # split_index = [int(shadow_trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    # split_index.append(int(shadow_trainset.__len__() - int(shadow_trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    # shadow_client_dataset = torch.utils.data.random_split(shadow_trainset, split_index)
    client_loaders = []
    # shadow_client_sloaders = []
    for ii in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))
        # shadow_client_loaders.append(DataLoader(shadow_client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))
        

    return client_loaders, test_loader

"""Function: load data"""
def data_init_with_shadow(FL_params):

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    whole_trainset, whole_testset = data_set(FL_params.data_name)
    shadow_split_idx = [int(whole_trainset.__len__()/2), int(whole_trainset.__len__()) -int(whole_trainset.__len__()/2)]
    trainset, shadow_trainset = torch.utils.data.random_split(whole_trainset, shadow_split_idx)

    shadow_split_idx = [int(whole_testset.__len__()/2), int(whole_testset.__len__()) -int(whole_testset.__len__()/2)]
    testset, shadow_testset = torch.utils.data.random_split(whole_testset, shadow_split_idx)


    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)
    shadow_test_loader = DataLoader(shadow_testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)


    split_index = [int(trainset.__len__()/FL_params.N_client)]*(FL_params.N_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_client)*(FL_params.N_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)

    split_index = [int(shadow_trainset.__len__()/FL_params.N_client)]*(FL_params.N_client-1)
    split_index.append(int(shadow_trainset.__len__() - int(shadow_trainset.__len__()/FL_params.N_client)*(FL_params.N_client-1)))
    shadow_client_dataset = torch.utils.data.random_split(shadow_trainset, split_index)
   
    client_loaders = []
    shadow_client_loaders = []
    for ii in range(FL_params.N_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))
        shadow_client_loaders.append(DataLoader(shadow_client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))

    return client_loaders, test_loader, shadow_client_loaders, shadow_test_loader

def data_set(data_name, data_dir):
    if not data_name in ['mnist', 'purchase', 'adult', 'cifar10']:
        raise TypeError('data_name should be a string, including mnist, purchase, adult, cifar10.')

    # Specify the root directory for storing/accessing datasets
    root_dir = os.path.join(data_dir, data_name)

    # Check if the directory exists, if not, create it
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Adjust the dataset loading paths to the root directory
    if data_name == 'mnist':
        trainset = datasets.MNIST(root=root_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        testset = datasets.MNIST(root=root_dir, train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    elif data_name == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)

        testset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    elif data_name == 'purchase':
        # Load purchase dataset
        xx = np.load(os.path.join(root_dir, "purchase_xx.npy"))
        yy = np.load(os.path.join(root_dir, "purchase_y2.npy"))

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)

        # Create datasets
        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)
    elif data_name == 'adult':
        # Load adult dataset
        file_path = os.path.join(root_dir, "adult")
        data1 = pd.read_csv(os.path.join(file_path, 'adult.data'), header=None)
        data2 = pd.read_csv(os.path.join(file_path, 'adult.test'), header=None)
        data2 = data2.replace(' <=50K.', ' <=50K')
        data2 = data2.replace(' >50K.', ' >50K')
        train_num = data1.shape[0]
        data = pd.concat([data1, data2])

        # Convert categorical features to numerical labels
        data = np.array(data, dtype=str)
        labels = data[:, 14]
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:, :-1]

        categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
        data = data.astype(float)

        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        for feature in numerical_features:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[:, feature].reshape(-1, 1))
            data[:, feature] = scaled_data.reshape(-1)

        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features)],
            remainder='passthrough')
        oh_data = oh_encoder.fit_transform(data)

        xx = oh_data
        yy = labels
        xx = preprocessing.scale(xx)
        yy = np.array(yy)

        xx = torch.Tensor(xx).type(torch.FloatTensor)
        yy = torch.Tensor(yy).type(torch.LongTensor)
        xx_train = xx[0:data1.shape[0], :]
        xx_test = xx[data1.shape[0]:, :]
        yy_train = yy[0:data1.shape[0]]
        yy_test = yy[data1.shape[0]:]

        trainset = TensorDataset(xx_train, yy_train)
        testset = TensorDataset(xx_test, yy_test)

    return trainset, testset


#define class->dataset  for adult and purchase datasets
#for the purchase, we use TensorDataset function to transform numpy.array to datasets class
#for the adult, we custom an AdultDataset class that inherits torch.util.data.Dataset class
"""
Array2Dataset: A class that can transform np.array(tensor matrix) to a torch.Dataset class.
"""
class Array2Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    def __getitem__(self, index):
        x = self.data[index,:]
        y = self.targets[index]
        return x, y
    def __len__(self):
        return len(self.data)