import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import StrMethodFormatter

###########################################################################################
# CONSTANTS 
###########################################################################################

algorithms = ["pFedMe_p", "pFedMe", "PerAvg_p", "FedAvg"]

dataset = "Mnist"
Numb_Glob_Iters = 800

params = {
    "results_DNN": {
        "learning_rates": [0.01, 0.01, 0.02, 0.02],
        "betas": [2.0, 2.0, 0.001, 1.0],
        "lambdas": [30, 30, 15, 15],
        "personal_learning_rate": [0.05, 0.05, 0.05, 0.05],
        "local_epochs": [20, 20, 20, 20],
        "K": [5, 5, 5, 5],
        "batch_sizes": [20, 20, 20, 20]
    },
    "results_MLR": {
        "learning_rates": [0.01, 0.01, 0.03, 0.02],
        "betas": [2.0, 2.0, 0.003, 1.0],
        "lambdas": [15, 15, 15, 15],
        "personal_learning_rate": [0.1, 0.1, 0.1, 0.1],
        "local_epochs": [20, 20, 20, 20],
        "K": [5, 5, 5, 5],
        "batch_sizes": [20, 20, 20, 20]
    }
}

###########################################################################################
# To read the h5 file 
###########################################################################################

def simple_read_data(alg, folder=""):
    """
    Read training accuracy, training loss, and global accuracy from an HDF5 file.

    Parameters:
    alg (str): The name of the algorithm.
    folder (str): The folder name where the HDF5 file is located.

    Returns:
    tuple: A tuple containing training accuracy, training loss, and global accuracy.
    """
    
    path = os.path.join(folder, '{}.h5'.format(alg))
    hf = h5py.File(path, 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc

###########################################################################################
# For the average 
###########################################################################################

def get_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],beta=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = [], folder=""):
    """
    Get training accuracy, training loss, and global accuracy data from HDF5 files.

    Parameters:
    num_users (int): Number of users.
    loc_ep1 (int): Number of local epochs.
    Numb_Glob_Iters (int): Number of global iterations.
    lamb (list): List of lambda values.
    learning_rate (list): List of learning rates.
    beta (list): List of beta values.
    algorithms_list (list): List of algorithm names.
    batch_size (list): List of batch sizes.
    dataset (str): Name of the dataset.
    k (list): List of k values.
    personal_learning_rate (list): List of personal learning rates.
    folder (str): The folder name where the HDF5 files are located.

    Returns:
    tuple: A tuple containing global accuracy, training accuracy, and training loss data.
    """

    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()

    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])  
        string_learning_rate = string_learning_rate + "_" +str(beta[i]) + "_" +str(lamb[i])
        if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(dataset +"_"+ algorithms_list[i] + "_avg", folder))[:, :Numb_Glob_Iters]
        algorithms_list[i] = algs_lbl[i]
    return glob_acc, train_acc, train_loss

def get_max_value_index(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],beta=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = [], folder=""):
    """
    Print the maximum testing accuracy and its index for each algorithm.

    Parameters:
    num_users (int): Number of users.
    loc_ep1 (int): Number of local epochs.
    Numb_Glob_Iters (int): Number of global iterations.
    lamb (list): List of lambda values.
    learning_rate (list): List of learning rates.
    beta (list): List of beta values.
    algorithms_list (list): List of algorithm names.
    batch_size (list): List of batch sizes.
    dataset (str): Name of the dataset.
    k (list): List of k values.
    personal_learning_rate (list): List of personal learning rates.
    folder (str): The folder name where the HDF5 files are located.
    """

    results = []

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate, folder)
    
    for i in range(Numb_Algs):
        results.append({
            "Algorithm": algorithms_list[i],
            "Folder": folder.split("/")[-1],
            "Max testing Accuracy": glob_acc[i].max(),
            "Index": np.argmax(glob_acc[i]),
            "Mean": np.mean(glob_acc[i]),
            "Std": np.std(glob_acc[i]),
        })

    return pd.DataFrame(results)

def max_average_df(num_users, folders):
    results_all = []

    for folder in folders:
        folder_name = folder.split("/")[-1]
        results_df = get_max_value_index(
            num_users=num_users,
            loc_ep1=params[folder_name]["local_epochs"],
            Numb_Glob_Iters=Numb_Glob_Iters,
            lamb=params[folder_name]["lambdas"],
            learning_rate=params[folder_name]["learning_rates"],
            beta=params[folder_name]["betas"],
            algorithms_list=algorithms,
            batch_size=params[folder_name]["batch_sizes"],
            dataset=dataset,
            k=params[folder_name]["K"],
            personal_learning_rate=params[folder_name]["personal_learning_rate"],
            folder=folder
        )
        results_all.append(results_df)
        #print(results_df)

    # Concatenate all DataFrames
    all_results_df = pd.concat(results_all, ignore_index=True)

    return all_results_df

def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len < 3:
        return data
    for i in range(len(data)):
        x = data[i]
        print(f"Original data for index {i}: {x}, {len(x)}")
        s=np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        #print(f"Extended data for index {i}: {s}")
        if window == 'flat': #moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.'+ window + '(window_len)')
        #print(f"Window for index {i}: {w}")

        y=np.convolve(w/w.sum(), s, mode='valid')
        #print(f"Smoothed data for index {i}: {y}")

        results.append(y[window_len-1:])
        #print(f"Final smoothed data for index {i}: {y[window_len-1:]}")
    return np.array(results)

###########################################################################################
# For all the runs 
###########################################################################################

def get_all_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0, beta=0, algorithms="", batch_size=0, dataset="", k=0, personal_learning_rate=0, times=10, folder=""):
    """
    Get training accuracy, training loss, and global accuracy data from HDF5 files.

    Parameters:
    num_users (int): Number of users.
    loc_ep1 (int): Number of local epochs.
    Numb_Glob_Iters (int): Number of global iterations.
    lamb (float): Lambda value.
    learning_rate (float): Learning rate.
    beta (float): Beta value.
    algorithms (str): Name of the algorithm.
    batch_size (int): Batch size.
    dataset (str): Name of the dataset.
    k (int): k value.
    personal_learning_rate (float): Personal learning rate.
    times (int): Number of times to repeat the experiment.
    folder (str): The folder name where the HDF5 files are located.

    Returns:
    tuple: A tuple containing global accuracy, training accuracy, and training loss data.
    """

    train_acc = np.zeros((times, Numb_Glob_Iters))
    train_loss = np.zeros((times, Numb_Glob_Iters))
    glob_acc = np.zeros((times, Numb_Glob_Iters))
    algorithms_list = [algorithms] * times

    for i in range(times):
        string_learning_rate = str(learning_rate)
        string_learning_rate = string_learning_rate + \
            "_" + str(beta) + "_" + str(lamb)
        if(algorithms == "pFedMe" or algorithms == "pFedMe_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + \
                str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" + \
                str(loc_ep1) + "_" + str(k) + "_" + \
                str(personal_learning_rate) + "_" + str(i)
        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + \
                str(num_users) + "u" + "_" + str(batch_size) + "b" + \
                "_" + str(loc_ep1) + "_" + str(i)

        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(dataset + "_" + algorithms_list[i], folder))[:, :Numb_Glob_Iters]
    return glob_acc, train_acc, train_loss


def get_max_value_index_all(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0, beta=0, algorithms="", batch_size=0, dataset="", k=0, personal_learning_rate=0, times=10, folder=""):
    """
    Get maximum testing accuracy, its index, mean accuracy, and variance for each experiment.

    Parameters:
    num_users (int): Number of users.
    loc_ep1 (int): Number of local epochs.
    Numb_Glob_Iters (int): Number of global iterations.
    lamb (float): Lambda value.
    learning_rate (float): Learning rate.
    beta (float): Beta value.
    algorithms (str): Name of the algorithm.
    batch_size (int): Batch size.
    dataset (str): Name of the dataset.
    k (int): k value.
    personal_learning_rate (float): Personal learning rate.
    times (int): Number of times to repeat the experiment.
    folder (str): The folder name where the HDF5 files are located.

    Returns:
    pandas.DataFrame: A DataFrame containing the results.
    """

    results = []
    max_accurancy = []
    glob_acc, _, _ = get_all_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms, batch_size, dataset, k, personal_learning_rate, times, folder)

    for i in range(times):
        results.append({
            "Algorithm": algorithms,
            "Folder": folder.split("/")[-1],
            "Max testing Accuracy": glob_acc[i].max(),
            "Index": np.argmax(glob_acc[i]),
        })
        max_accurancy.append(glob_acc[i].max())
    mean = np.mean(max_accurancy)
    std = np.std(max_accurancy)

    df = pd.DataFrame(results)

    df["Mean Max testing Accuracy"] = mean
    df["Std Max testing Accuracy"] = std

    return df

def max_df(num_users, folders):
    results_all = []

    for folder in folders:
        folder_name = folder.split("/")[-1]
        for i in range(len(algorithms)):
            results_df = get_max_value_index_all(
                num_users=num_users,
                loc_ep1=params[folder_name]["local_epochs"][i],
                Numb_Glob_Iters=Numb_Glob_Iters,
                lamb=params[folder_name]["lambdas"][i],
                learning_rate=params[folder_name]["learning_rates"][i],
                beta=params[folder_name]["betas"][i],
                algorithms=algorithms[i],
                batch_size=params[folder_name]["batch_sizes"][i],
                dataset=dataset,
                folder=folder,
                k=params[folder_name]["K"][i],
                personal_learning_rate=params[folder_name]["personal_learning_rate"][i]
            )
            results_all.append(results_df)
            #print(results_all)

    # Concatenate all DataFrames
    all_results_df = pd.concat(results_all, ignore_index=True)

    # Get only the rows with maximum accuracy for each algorithm and each folder
    max_results_df = all_results_df.loc[all_results_df.groupby(["Algorithm", "Folder"])["Max testing Accuracy"].idxmax()]
    '''
    # Get the mean and variance for each algorithm and each folder
    mean_var_results_df = all_results_df.groupby(["Algorithm", "Folder"]).agg(
        {"Max testing Accuracy": ["mean", "var"]}
    )
    # Rename columns
    mean_var_results_df.columns = ["Mean Accuracy", "Variance"]

    # Reset index
    mean_var_results_df = mean_var_results_df.reset_index()

    # Merge with max_results_df
    max_results_df = max_results_df.merge(mean_var_results_df, on=["Algorithm", "Folder"])
    '''
    return max_results_df

###########################################################################################
# For the heatmaps
###########################################################################################

def heatmaps(data_dir):
    DATA_DIR = data_dir
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

###########################################################################################
# For the figure 6 (compare the two models MCLR and DNN)
###########################################################################################

def get_label_name(name):
    if name.startswith("pFedMe"):
        if name.startswith("pFedMe_p"):
            return "pFedMe"+ " (PM)"
        else:
            return "pFedMe"+ " (GM)"
    if name.startswith("PerAvg"):
        return "Per-FedAvg"
    if name.startswith("FedAvg"):
        return "FedAvg"
    if name.startswith("APFL"):
        return "APFL"

def plot_summary_one_figure_mnist_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate, folder, y_lim):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    folder_path = folder[2:]
    folder_name = folder.split("/")[-1]
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate, folder)
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    print(f"Shape of glob_acc_: {glob_acc.shape}")
    print(f"Shape of train_acc_: {train_acc.shape}")
    print(f"Shape of train_loss_: {train_loss.shape}")
    
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1, figsize=(5, 5))
    plt.title(folder_name)
    plt.grid(True)
    # training loss
    for i in range(Numb_Algs):
        print(f"Algorithm {i}: {algorithms_list[i]}, Data length: {len(train_loss[i, 1:])}")
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    if y_lim:
        plt.ylim([0,  0.6])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    #plt.savefig(folder_path +  "_train_"+ '_'.join(folder_path.split('/')[:-1]) + ".png", bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(2, figsize=(5, 5))
    plt.title(folder_name)
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        print(f"Algorithm {i}: {algorithms_list[i]}, Data length: {len(glob_acc[i, 1:])}")
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    if y_lim:
        plt.ylim([0.84,  0.98]) # non convex-case
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    #plt.savefig(folder_path + "_test_"+ '_'.join(folder_path.split('/')[:-1]) + ".png", bbox_inches="tight")
    plt.show()
    plt.close()

def plot_comparison(num_users, folders, y_lim=True):
    for folder in folders:
        folder_name = folder.split("/")[-1]
        print(folder)
        plot_summary_one_figure_mnist_Compare(num_users=num_users,
                                            loc_ep1=params[folder_name]["local_epochs"],
                                            Numb_Glob_Iters=Numb_Glob_Iters,
                                            lamb=params[folder_name]["lambdas"],
                                            learning_rate=params[folder_name]["learning_rates"],
                                            beta = params[folder_name]["betas"],
                                            algorithms_list=algorithms,
                                            batch_size=params[folder_name]["batch_sizes"],
                                            dataset=dataset,
                                            folder=folder,
                                            k = params[folder_name]["K"],
                                            personal_learning_rate = params[folder_name]["personal_learning_rate"],
                                            y_lim = y_lim)
