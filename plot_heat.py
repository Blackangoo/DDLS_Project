import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
#DATA_DIR = 'dirichlet_datasets/with_shuffle/mnist_train_D1.json'
def main(DATA_DIR):
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
    plt.savefig(DATA_DIR.replace('.json', '.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, help="Path to the JSON data file")
    args = parser.parse_args()

    if args.DATA_DIR:
        main(args.DATA_DIR)
    else:
        print("Please provide the path to the JSON data file using --DATA_DIR option.")