import matplotlib.pyplot as plt
import json
import numpy as np


DATA_DIR = 'data/Mnist/data/train/mnist_train_pFedMe_D0_2.json'
# Load the JSON file
with open(DATA_DIR) as f:
    train_data = json.load(f)

# Extract users and tags
users = train_data['users']

tags = [str(i) for i in range(10)]  # Assuming tags are from 0 to 9

# Create a matrix to store tag frequencies for each user
matrix = np.zeros((len(tags), len(users)), dtype=int)
# Populate the matrix with tag frequencies
for idx, user_id in enumerate(users):
    user_tags = train_data['user_data'][user_id]['y']
    # print(user_id)
    # print(user_tags)
    # print('------------------------------')
    for tag in user_tags:
        matrix[int(tag), idx] += 1  # Convert tag to int before indexing
users_display = [user_id[-3:] for user_id in users] #Shorten user name
# Plot heatmap
print(matrix)
plt.figure(figsize=(10, 8))
plt.imshow(matrix, cmap='hot', interpolation='nearest', aspect='auto')
plt.colorbar(label='Frequency')
plt.title('Tag Frequency Heatmap')
plt.xlabel('User')
plt.ylabel('Tag')
plt.xticks(range(len(users)), users_display, rotation=45)
plt.yticks(range(len(tags)), tags)
plt.tight_layout()
plt.show()