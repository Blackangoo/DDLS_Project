# Personalized Federated Learning with Moreau Envelopes
This repository implements all experiments in the paper [**Personalized Federated Learning with Moreau Envelopes**](https://arxiv.org/pdf/2006.08848) - Canh T. Dinh, Nguyen H. Tran, Tuan Dung Nguyen and extends it with an investigation of the performance and robustness of the pFedMe algorithm under different data distributions and attack scenarios.

This repository does not only implement pFedMe but also FedAvg, and Per-FedAvg algorithms.

# Software requirements:
- numpy, scipy, torch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**
  
# Dataset: We use 1 dataset: MNIST
- To generate non-idd MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_20users.py"
- To generate non-iid MNIST Data based on a Dirichlet distribution:
  - Access data/Mnist and run: "python3 generate_niid_20users_dirichlet.py"

- The datasets also are available to download at: https://drive.google.com/drive/folders/1mzdFPHgYjjvR8Yspfv3WxfGK855Z41Ak

# Produce experiments and figures

- There is a main file "main.py" which allows running all experiments.
  
- It is noted that each algorithm should be run at least 10 times and then the results are averaged.

- All the train loss, testing accuracy, and training accuracy will be stored as h5py file in the folder "results". It is noted that we store the data for persionalized model and global of pFedMe in 2 separate files following format: DATASET_pFedMe_p_x_x_xu_xb_x_avg.h5 and DATASET_pFedMe_x_x_xu_xb_x_avg.h5 respectively (pFedMe for global model, pFedMe_p for personalized model of pFedMe, PerAvg_p is for personalized model of PerAvg).

## Heatmaps
To visualize the different distribution of the labels per clients:
<p align="center">
  <img src="https://raw.githubusercontent.com/Blackangoo/DDLS_Project/main/results_images/datasets/baseline_train.png">
</p>

## Fine-tuned Parameters:
To produce results in the table of fine-tune parameter:

<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/83839182-9fdb9400-a73e-11ea-8416-9cdfcecacd75.png">
</p>


- MNIST:
  - Strongly Convex Case:
    <pre><code>
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.1 --beta 2 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 --times 10
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.02 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5 --times 10
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.003  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5 --times 10
    </code></pre>
  
  - NonConvex Case:
    <pre><code>
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 --times 10
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.02 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5 --times 10
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.02 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5 --times 10
    </code></pre>

## Attacks:
To produce the results for the attacks:
- MNIST:
  - Strongly Convex Case:
    <pre><code>
    python3 attacks.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.1 --beta 2 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 --times 10
    python3 attacks.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.02 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5 --times 10
    python3 attacks.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.003  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5 --times 10
    </code></pre>
  
  - NonConvex Case:
    <pre><code>
    python3 attacks.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 --times 10
    python3 attacks.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.02 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5 --times 10
    python3 attacks.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.02 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5 --times 10
    </code></pre>
