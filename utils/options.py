#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
from Setting import DATASET, LOCAL_EPOCH, NUM_GLOBAL_ITERS, RUNNING_ALG, Frac_users, N_clients, local_learning_rate


def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    if(DATASET == "mnist"):
        Input_DS = "Mnist"
    else:
        Input_DS = DATASET
    if(RUNNING_ALG == "fedavg"):
        Input_Alg = "FedAvg"
    else:
        Input_Alg = RUNNING_ALG

    parser.add_argument("--dataset", type=str, default=Input_DS, choices=[
                        "EMNIST", "human_activity", "gleam", "vehicle_sensor", "mnist", "Synthetic", "Cifar10", "fmnist", "Cifar100"])
    # parser.add_argument("--server_model", type=str, default="cnn", choices=["cnn","resnet"])
    parser.add_argument("--client_model", type=str,
                        default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float,
                        default=local_learning_rate, help="Local learning rate")
    parser.add_argument("--L_k", type=float, default=1,
                        help="Regularization term")
    parser.add_argument("--num_global_iters", type=int,
                        default=NUM_GLOBAL_ITERS)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCH)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default=Input_Alg, choices=[
                        "pFedMe", "pFedMe_p", "PerAvg", "FedAvg", "FedU", "Mocha", "Local", "Global", "DemLearn", "DemLearnRep", "CDKT"])
    parser.add_argument("--subusers", type=float, default=Frac_users,
                        help="Fraction of Num Users per round")  # Fraction number of users
    parser.add_argument("--K", type=int, default=0, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.02,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--comet", type=int, default=0,
                        help="log data to comet")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments")  # GPU dev_id, -1 is CPU
    parser.add_argument("--cutoff", type=int, default=0,
                        help="Cutoff data sample")
    parser.add_argument("--beta", type=float, default=0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--DECAY", type=bool, default=0,
                        help="DECAY or CONSTANT")
    parser.add_argument("--mu", type=int, default=0, help="mu parameter")
    parser.add_argument("--gamma", type=int, default=0, help="gama parameter")
    parser.add_argument("--total_users", type=int,
                        default=N_clients, help="total users")
    parser.add_argument("--K_Layer_idx", nargs="*", type=int,
                        default=0, help="Model Layer Index")
    args = parser.parse_args()

    return args
