#!/usr/bin/env python

# import comet_ml at the top of your file
from comet_ml import Experiment
from utils.options import args_parser
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverFedU import FedU
from FLAlgorithms.servers.serverlocal import FedLocal
from FLAlgorithms.servers.serverglobal import FedGlobal
from FLAlgorithms.servers.serverCDKT import CDKT
from utils.model_utils import read_data
from FLAlgorithms.trainmodel.models import DNN, DNN2, CNNCifar, CNNCifar_Server, CNNCifar_Server_3layer, Mclr_Logistic, Net_DemAI, Net_DemAI_Client
import torch

from utils.plot_utils import average_data
torch.manual_seed(0)


# python CDKT_main.py --dataset Mnist --model cnn --learning_rate 0.03 --num_global_iters 200  --algorithm FedAvg --times 1 --subusers 0.1
# python CDKT_main.py --dataset Mnist --model cnn --learning_rate 0.03 --num_global_iters 200  --algorithm --times 1 --subusers 0.1

# python CDKT_main.py --dataset Mnist --model cnn --learning_rate 0.03 --num_global_iters 200  --algorithm --times 1 --subusers 1


# Create an experiment with your api key:
def main(experiment, dataset, algorithm, model,  client_model, batch_size, learning_rate, beta, L_k, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, comet, gpu, cutoff, args):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(
        gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    data = read_data(dataset), dataset

    for i in range(times):
        print("---------------Running time:------------", i)
        # Generate model
        if(model == "mclr" or client_model == "mclr"):
            if(dataset == "human_activity"):
                model = Mclr_Logistic(561, 6).to(device), model
            elif(dataset == "gleam"):
                model = Mclr_Logistic(561, 6).to(device), model
            elif(dataset == "vehicle_sensor"):
                model = Mclr_Logistic(100, 2).to(device), model
            elif(dataset == "Synthetic"):
                model = Mclr_Logistic(60, 10).to(device), model
            elif(dataset == "EMNIST"):
                model = Mclr_Logistic(784, 62).to(device), model
            else:  # (dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model

        elif(model == "dnn" or client_model == "dnn"):
            if(dataset == "human_activity"):
                model = DNN(561, 100, 12).to(device), model
            elif(dataset == "gleam"):
                model = DNN(561, 20, 6).to(device), model
            elif(dataset == "vehicle_sensor"):
                model = DNN(100, 20, 2).to(device), model
            elif(dataset == "Synthetic"):
                model = DNN(60, 20, 10).to(device), model
            elif(dataset == "EMNIST"):
                model = DNN(784, 200, 62).to(device), model
            else:  # (dataset == "Mnist"):
                model = DNN2().to(device), model

        elif(model == "cnn" or client_model == "cnn"):
            if (dataset == "EMNIST"):
                model = Net_DemAI().to(device), model
            elif (dataset == "Cifar10"):
                # model = CNNCifar_Server_3layer(10).to(device), model
                model = CNNCifar_Server(10).to(device), model
                # server_model = CNNCifar_Server(10).to(device)
                client_model = CNNCifar(10).to(device)
            elif (dataset == "Cifar100"):
                model = CNNCifar_Server_3layer(100).to(device), model
                # model = CNNCifar_Server(10).to(device), model
                # server_model = CNNCifar_Server(10).to(device)
                client_model = CNNCifar(100).to(device)
            else:  # (dataset == "Mnist"):
                model = Net_DemAI().to(device), model
                # server_model = Net_DemAI().to(device)
                client_model = Net_DemAI_Client().to(device)

        # select algorithm
        if(algorithm == "FedAvg"):
            if(comet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers) + "_" + str(args.total_users))
            server = FedAvg(experiment, device, data, algorithm, model, client_model, batch_size,
                            learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff, args)

        elif(algorithm == "PerAvg"):
            if(comet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(
                    personal_learning_rate) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers) + "_" + str(args.total_users))
            server = PerAvg(experiment, device, data, algorithm, model, batch_size, learning_rate,
                            beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff)

        elif (algorithm == "CDKT"):
            if (comet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers) + "_" + str(args.total_users))
            server = CDKT(experiment, device, data,  algorithm, model, client_model, batch_size,
                          learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff, args)

        elif(algorithm == "FedU"):
            if(comet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(L_k) + "L_K" + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers) + "_" + str(args.total_users))
            server = FedU(experiment, device, data, algorithm, model, batch_size, learning_rate,
                          beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, K, i, cutoff)

        elif(algorithm == "pFedMe"):
            if(comet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(
                    personal_learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers) + "_" + str(args.total_users))
            server = pFedMe(experiment, device, data, algorithm, model, batch_size, learning_rate, beta,
                            L_k, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, cutoff)

        elif(algorithm == "Local"):
            if(comet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(L_k) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers) + "_" + str(args.total_users))
            server = FedLocal(experiment, device, data, algorithm, model, batch_size, learning_rate,
                              beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff)

        elif(algorithm == "Global"):
            if(comet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(L_k) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers) + "_" + str(args.total_users))
            server = FedGlobal(experiment, device, data, algorithm, model, batch_size, learning_rate,
                               beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff)
        else:
            print("Algorithm is invalid")
            return
    if comet:
        with experiment.train():
            server.train()
    else:
        server.train()

    # average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L_k, learning_rate=learning_rate,
    #              beta=beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k=K, personal_learning_rate=personal_learning_rate,
    #              times=times, cutoff=cutoff)


if __name__ == "__main__":
    args = args_parser()
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("num of users       : {}".format(args.total_users))
    print("Subset of users      : {}".format(args.subusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    # print("Server Model       : {}".format(args.server_model))
    print("Client Model       : {}".format(args.client_model))
    print("=" * 80)

    if(args.comet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="eOnZ4ncwzyZMH3k7s7YzEGIQi",
            project_name="test-project",
            workspace="jevenm",
        )

        hyper_params = {
            "dataset": args.dataset,
            "algorithm": args.algorithm,
            "model": args.model,
            # "server_model":args.server_model,
            "client_model": args.client_model,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "L_k": args.L_k,
            "num_glob_iters": args.num_global_iters,
            "local_epochs": args.local_epochs,
            "optimizer": args.optimizer,
            "numusers": args.subusers,
            "totalusers": args.total_users,
            "K": args.K,
            "personal_learning_rate": args.personal_learning_rate,
            "times": args.times,
            "gpu": args.gpu,
            "cut-off": args.cutoff
        }

        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment=experiment,
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        # server_model=args.server_model,
        client_model=args.client_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        L_k=args.L_k,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        numusers=args.subusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        comet=args.comet,
        gpu=args.gpu,
        cutoff=args.cutoff,
        args=args
    )
