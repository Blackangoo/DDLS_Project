# Import necessary libraries
from sklearn.datasets import fetch_openml
from tqdm import trange  # Progress bar
import numpy as np  # Numerical operations
import numpy.random as npr
import random  # Random number generation
import json  # JSON serialization/deserialization
import os  # Operating system operations
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from typing import cast


# Set random seeds for reproducibility
random.seed(1)
npr.seed(1)

# Define the number of users and labels
NUM_USERS = 20  # Should be multiple of 10
NUM_LABELS = 2  # num of labels each user will get
ALPHA = 0.2

# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Load Mnist dataset
data_path_str = "./data"
ETA = "\N{GREEK SMALL LETTER ETA}"
device = torch.device("cpu")
torch.backends.cudnn.deterministic=True

transform = transforms.Compose([
    transforms.ToTensor(),
    # normalize by training set mean and standard deviation
    # resulting data has mean=0 and std=1
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(data_path_str, download=True, transform=transform)

# Dirichlet split
rng = npr.default_rng(1)
client_alphas = np.repeat(ALPHA, NUM_USERS)

permutation_indices = rng.permutation(len(train_dataset))
permuted_targets = np.array([target for _data, target in train_dataset])[permutation_indices]
splits = [[] for _ in range(NUM_USERS)]

for label in np.unique(permuted_targets):
    label_indices_original = permutation_indices[np.nonzero(permuted_targets == label)]
    label_split_indices = (
        rng.dirichlet(client_alphas).cumsum() * len(label_indices_original)
    ).astype(int)
    label_splits = np.split(label_indices_original, label_split_indices[:-1])

    for split, label_split in zip(splits, label_splits):
        split.extend(label_split)

final_splits =  [Subset(train_dataset, split) for split in cast(list[list[int]], splits)]


# Formatting the data to store in a json file
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]

for user in trange(NUM_USERS):
    for split in final_splits[user]: 
        # Splits are tuples with the values of the pixels (as tensors) and the target value
        X[user].append(split[0].flatten().numpy().tolist())
        y[user].append(split[1])

    print(f"\nUser {user}: done, num samples= {len(X[user])}")
    print(f"Y : {y[user][0:3]}")


print(sum([len(X[i]) for i in range(NUM_USERS)]))
print(len(y), len(y[0]))
# From here it is the same as generate_niid_20users.py
# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.75*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))

# Save generated samples
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")

