# Import necessary libraries
from sklearn.datasets import fetch_openml
from tqdm import trange  # Progress bar
import numpy as np  # Numerical operations
import random  # Random number generation
import json  # JSON serialization/deserialization
import os  # Operating system operations

# Set random seeds for reproducibility
random.seed(1)
np.random.seed(1)

# Define the number of users and labels
NUM_USERS = 20  # Should be multiple of 10
NUM_LABELS = 2  # num of labels each user will get

# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
mnist = fetch_openml('mnist_784', data_home='./data')
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)

mnist_data = []
for i in trange(10):
    idx = mnist.target==str(i) # a boolean array indicating which MNIST data corresponds to the current label i
    mnist_data.append(mnist.data[idx]) # Adds data corresponding to label i to mnist_data
    print("\nNumber of samples for label", i, ":", np.sum(idx))  # Prints the number of samples for each label

# commented those cause I don't know their utility here
# print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
# print("idx",idx)

users_labels = []

# the labels will be distributed in a cyclic manner such that each user gets a unique combination of 2 labels.
# The users_labels list will contain all the assigned labels
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):
        l = (user * NUM_USERS + j) % 10
        #l = (user * NUM_LABELS + j) % 10
        users_labels.append(l)

print(f"user_labels = {users_labels}")
unique, counts = np.unique(users_labels, return_counts=True)
print("42--------------")
print(f"unique= {unique}, counts={counts}")

# Function to generate random numbers for label distribution
def ram_dom_gen(total, size):
    print(total, size)
    temp = []
    for _ in range(size - 1):
        val = np.random.randint(total//(size + 1), total//2)
        #val = 100
        temp.append(val)
        total -= val
    temp.append(total)
    print(temp)
    return temp

# Generate random number of samples for each label and user
number_sample = []
for total_value, count in zip(mnist_data, counts):
    print(f"total_value= {total_value}, count= {count}")
    temp = ram_dom_gen(len(total_value), count)
    number_sample.append(temp)
print("--------------")
print(f"number_sample= {number_sample}")

# Arrange generated samples for each label and user
i = 0
number_samples = []
for i in range(len(number_sample[0])):
    for sample in number_sample:
        print(sample)
        number_samples.append(sample[i])

print("--------------")
print(f"number_samples= {number_samples}")

###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)
count = 0
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user + j) % 10
        #l = (user * NUM_LABELS + j) % 10
        print("value of L",l)
        print("value of count",count)
        num_samples =  number_samples[count] # num sample
        count = count + 1
        if idx[l] + num_samples < len(mnist_data[l]):
        #if idx[l] + num_samples <= len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].values.tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
            print("check len os user:", user, j,"len data", len(X[user]), num_samples)

print("IDX2:", idx) # counting samples for each labels
print(len(y), len(y[0]))
# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
print(f"X [19] : {len(X[19])}")
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

print(f"Test data num sample: {test_data['num_samples']}")

# Save generated samples
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
