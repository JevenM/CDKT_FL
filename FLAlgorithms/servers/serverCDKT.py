import torch
from tqdm import tqdm
import torch.nn as nn
from FLAlgorithms.users.userCDKT import UserCDKT
from FLAlgorithms.servers.serverbase_dem import Dem_Server
from Setting import rs_file_path, N_clients
from utils.data_utils import write_file
from utils.dem_plot import plot_from_file
from utils.model_utils import read_user_data, read_public_data
from torch.utils.data import DataLoader
import numpy as np
from FLAlgorithms.optimizers.fedoptimizer import DemProx_SGD
from FLAlgorithms.trainmodel.models import *
# Implementation for Server
from utils.train_utils import KL_Loss, JSD
from Setting import *


class CDKT(Dem_Server):
    '''
    @Mao
    '''

    def __init__(self, experiment, device, dataset, algorithm, model,  client_model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, num_users, times, cutoff, args):
        super().__init__(experiment, device, dataset, algorithm,
                         model[0],  client_model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, num_users, times, args)

        # Initialize data for all  users
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = DemProx_SGD(
            self.model.parameters(), lr=global_learning_rate, mu=0)
        # when mu=0, the optimizer above is equivalent to the following line
        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=global_learning_rate)
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()
        # average server local output dict, key: batch_index
        self.avg_local_dict_prev_1 = dict()
        # TODO whats this
        self.gamma = gamma

        # default 0
        self.sub_data = cutoff
        if(self.sub_data):
            randomList = self.get_partion(self.total_users)

        # no shuffle
        public_dataset = read_public_data(dataset[0], dataset[1])
        # Mnist public len: 355
        print("public dataset len", len(public_dataset))

        self.publicdatasetloader = DataLoader(
            public_dataset, self.batch_size, shuffle=True)
        # self.publicloader= list(enumerate(self.publicdatasetloader))
        self.publicloader = []
        for b, (x, y) in enumerate(self.publicdatasetloader):
            self.publicloader.append((b, (x, y)))
            # y of a batch(20): tensor([4, 6, 2, 3, 8, 4, 2, 3, 3, 9, 3, 7, 4, 5, 4, 7, 5, 3, 6, 7])
            # if(b < 1):
            #     print(f"print y {y}")

        sample = []
        for i in range(self.total_users):
            # TODO 这一步的public可以提出来，把函数中的read_public_data重叠部分删除
            id, train, test, public = read_user_data(i, dataset[0], dataset[1])
            print("User ", id, ": Number of Train data", len(
                train), " Number of test data", len(test), " public len", len(public))
            sample.append(len(train)+len(test))

            # default 0
            if(self.sub_data):
                if(i in randomList):
                    train, test = self.get_data(train, test)
            # new user object
            user = UserCDKT(device, id, train, test, public, model, client_model,
                            batch_size, learning_rate, beta, local_epochs)
            # every user has the same loader
            user.publicdatasetloader = self.publicloader
            self.users.append(user)
            # calculate total samples for training
            self.total_train_samples += user.train_samples
            # print("user train samples == len(train): ", user.train_samples)
        # Mnist 73.5
        print(
            f"median of selected users samples: {sample} (sample) is", np.median(sample))

        print("Fraction number of users / total users:",
              num_users, " / ", self.total_users)

        print("Finished creating server.")

    def send_grads(self):
        '''
        @Mao

        Send grad to all user.

        '''
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def generalized_knowledge_construction(self, epochs, glob_iter):
        '''
        @Mao

        Construct the generalized knowledge and optimize the global model on public dataset `epochs` times.

        Args:
            `epochs`(int): global generalized epochs.
            `glob_iter`(int): global round index.
        '''
        self.model.train()

        avg_local_dict = dict()
        avg_rep_local_dict = dict()

        # key: batch_index, value: batch of local model's output on public data
        local_dict = dict()
        # key: batch_index, value: batch of local model's representation on public data
        rep_local_dict = dict()

        agg_local_dict = dict()
        # local_dict_prev = dict()

        clients_local_dict = dict()
        clients_rep_local_dict = dict()

        # user index
        c = 0
        for user in self.selected_users:
            c += 1
            local_dict.clear()
            for batch_idx, (X_public, y_public) in self.publicloader:
                X_public, y_public = X_public.to(
                    self.device), y_public.to(self.device)

                if Same_model:
                    local_output_public, rep_local_output_public = user.model(
                        X_public)
                else:
                    local_output_public, rep_local_output_public = user.client_model(
                        X_public)

                rep_local_output_public = rep_local_output_public.cpu().detach().numpy()
                local_output_public = local_output_public.cpu().detach().numpy()

                local_dict[batch_idx] = local_output_public
                rep_local_dict[batch_idx] = rep_local_output_public
            # print("local dict: ", local_dict)
            clients_local_dict[c] = local_dict
            clients_rep_local_dict[c] = rep_local_dict

        # dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # print("clients_rep_local_dict keys(client index): ", clients_rep_local_dict.keys())

        # Avg local output
        n = 0
        for client_idx in clients_local_dict.keys():
            # logits of a selected user
            c_logits = clients_local_dict[client_idx]

            # for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
            for batch_idx, (X_public, y_public) in self.publicloader:
                if (n == 0):
                    # TODO why divide by total_users not selected_users
                    avg_local_dict[batch_idx] = c_logits[batch_idx] / \
                        self.total_users
                else:
                    avg_local_dict[batch_idx] += c_logits[batch_idx] / \
                        self.total_users

            n += 1
        # dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        # print("avg_local_dict keys(batch_index): ", avg_local_dict.keys())

        # gamma ascending after more than 30 rounds
        if self.gamma < 0.8 and glob_iter >= 30:
            self.gamma += (0.8-0.5)/(NUM_GLOBAL_ITERS-30)
        print("averaging parameter gamma: ", self.gamma)

        print("avg_local_dict_prev_1 keys(should be empty): ",
              self.avg_local_dict_prev_1.keys())

        # moving average, average last round and this round by gamma
        for batch_idx, (X_public, y_public) in self.publicloader:
            # only 1st global round
            if (glob_iter == 0):
                agg_local_dict[batch_idx] = avg_local_dict[batch_idx]
            else:
                agg_local_dict[batch_idx] = self.gamma*self.avg_local_dict_prev_1[batch_idx] + (
                    1-self.gamma)*avg_local_dict[batch_idx]
        # save for last round
        self.avg_local_dict_prev_1 = agg_local_dict
        # dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        # print("agg_local_dict keys(batch_index): ", agg_local_dict.keys())

        # Avg rep local output according to num of users, num of samples of each user
        m = 0
        for client_idx in clients_rep_local_dict.keys():
            # a client
            c_rep_logits = clients_rep_local_dict[client_idx]

            for batch_idx, _ in self.publicloader:
                if (m == 0):
                    avg_rep_local_dict[batch_idx] = c_rep_logits[batch_idx] / \
                        self.total_users
                else:
                    avg_rep_local_dict[batch_idx] += c_rep_logits[batch_idx] / \
                        self.total_users
            m += 1

        # Avg rep local output according to num of samples of each user
        # k = 0
        # for client_idx in clients_rep_local_dict_2.keys():
        #
        #     c_rep_logits_2 = clients_rep_local_dict_2[client_idx]
        #     # print(self.total_train_samples)
        #
        #     for batch_idx, (X_public, y_public) in self.publicloader:
        #
        #         if (k == 0):
        #             avg_rep_local_dict[batch_idx] = c_rep_logits_2[batch_idx] / self.total_train_samples
        #         else:
        #             avg_rep_local_dict[batch_idx] += c_rep_logits_2[batch_idx] / self.total_train_samples
        #
        #     k += 1

        # Global distillation
        # implement several global iterations to construct generalized knowledge
        for _ in range(1, epochs+1):
            self.model.train()
            for batch_idx, (X_public, y_public) in self.publicloader:
                if Moving_Average:
                    batch_logits = torch.from_numpy(
                        agg_local_dict[batch_idx]).float().to(self.device)
                else:
                    batch_logits = torch.from_numpy(
                        avg_local_dict[batch_idx]).float().to(self.device)
                # NOTE why not moving average like agg_local_dict?
                batch_rep_logits = torch.from_numpy(
                    avg_rep_local_dict[batch_idx]).float().to(self.device)
                X_public, y_public = X_public.to(
                    self.device), y_public.to(self.device)

                output_public, rep_output_public = self.model(X_public)
                # torch.Size([20, 512]), last batch: torch.Size([15, 512])
                print("batch_rep_logits--: ", batch_rep_logits.shape)
                # torch.Size([20, 10]), last batch: torch.Size([15, 10])
                print("output_pub--", output_public.shape)

                if Tune_output:
                    y_onehot = F.one_hot(y_public, num_classes=NUMBER_LABEL)
                    batch_logits = (batch_logits + y_onehot)/2.0

                lossTrue = self.loss(output_public, y_public)
                lossKD = lossJSD = norm2loss = 0
                if Full_model:
                    lossKD = self.criterion_KL(output_public, batch_logits)
                    norm2loss = torch.dist(output_public, batch_logits, p=2)
                    lossJSD = self.criterion_JSD(output_public, batch_logits)
                else:
                    if Rep_Full:
                        lossKD = self.criterion_KL(output_public, batch_logits)
                        norm2loss = torch.dist(
                            output_public, batch_logits, p=2)
                        lossJSD = self.criterion_JSD(
                            output_public, batch_logits)
                        lossKD += self.criterion_KL(rep_output_public,
                                                    batch_rep_logits)
                        lossJSD += self.criterion_JSD(
                            rep_output_public, batch_rep_logits)
                        norm2loss = norm2loss + \
                            torch.dist(rep_output_public,
                                       batch_rep_logits, p=2)
                    else:  # only representation info
                        lossKD = self.criterion_KL(
                            rep_output_public, batch_rep_logits)
                        lossJSD = self.criterion_JSD(
                            rep_output_public, batch_rep_logits)
                        norm2loss = torch.dist(
                            rep_output_public, batch_rep_logits, p=2)

                if Global_CDKT_metric == "KL":
                    loss = lossTrue + beta * lossKD
                elif Global_CDKT_metric == "Norm2":
                    loss = lossTrue + beta * norm2loss
                elif Global_CDKT_metric == "JSD":
                    loss = lossTrue + beta * lossJSD

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

    def generalized_knowledge_ensemble(self, epochs):
        '''
        @Mao

        Simplified version generalized_knowledge_construction.

        Ensemble knowledge from selected clients to build generalized knowledge on server. Then optimize the global model `epochs` times on public dataset.

        Args:
            `epochs`(int): global generalized epochs

        '''
        self.model.train()

        local_dict = dict()
        rep_local_dict = dict()
        clients_local_dict = dict()
        clients_rep_local_dict = dict()

        # client index of selected users
        c = 0
        # clients predict on public data
        for user in self.selected_users:
            c += 1
            local_dict.clear()
            for batch_idx_public, (X_public, y_public) in self.publicloader:
                X_public, y_public = X_public.to(
                    self.device), y_public.to(self.device)

                if Same_model:
                    local_output_public, rep_local_output_public = user.model(
                        X_public)
                else:
                    local_output_public, rep_local_output_public = user.client_model(
                        X_public)

                rep_local_output_public = rep_local_output_public.cpu().detach().numpy()
                local_output_public = local_output_public.cpu().detach().numpy()

                local_dict[batch_idx_public] = local_output_public
                rep_local_dict[batch_idx_public] = rep_local_output_public
            clients_local_dict[c] = local_dict
            clients_rep_local_dict[c] = rep_local_dict

        # server
        for epoch in range(1, epochs+1):
            self.model.train()

            for batch_idx_public, (X_public, y_public) in self.publicloader:
                X_public, y_public = X_public.to(
                    self.device), y_public.to(self.device)
                if batch_idx_public == 0:
                    print('global batch[0] data: ', X_public[0])

                output_public, rep_output_public = self.model(X_public)
                print("output_public of server model: ", output_public)

                lossTrue = self.loss(output_public, y_public)
                lossKD = lossJSD = norm2loss = 0
                # Traverse the key of dictionary clients_rep_local_dict
                for client_index in clients_rep_local_dict:
                    # representation of a client on a batch
                    client_rep = clients_rep_local_dict[client_index][batch_idx_public]
                    client_batch_rep = torch.from_numpy(
                        client_rep).float().to(self.device)
                    # logits of a client on a batch
                    client_logits = clients_local_dict[client_index][batch_idx_public]
                    client_batch_logits = torch.from_numpy(
                        client_logits).float().to(self.device)

                    # only use output
                    if Full_model:
                        lossKD += self.criterion_KL(output_public,
                                                    client_batch_logits).to(self.device)
                        lossJSD += self.criterion_JSD(output_public,
                                                      client_batch_logits)
                        norm2loss += torch.dist(output_public,
                                                client_batch_logits, p=2)

                    else:
                        lossKD += self.criterion_KL(output_public,
                                                    client_batch_logits).to(self.device)
                        lossJSD += self.criterion_JSD(output_public,
                                                      client_batch_logits)
                        norm2loss += torch.dist(output_public,
                                                client_batch_logits, p=2)

                        # add representation info to loss
                        lossKD += self.criterion_KL(rep_output_public,
                                                    client_batch_rep).to(self.device)
                        lossJSD += self.criterion_JSD(
                            rep_output_public, client_batch_rep)
                        norm2loss += torch.dist(rep_output_public,
                                                client_batch_rep, p=2)

                if Global_CDKT_metric == "KL":
                    loss = lossTrue + beta * lossKD
                    # TODO Remove loss item losstrue, may have no labeled data
                    # loss =  beta * lossKD
                elif Global_CDKT_metric == "Norm2":
                    loss = lossTrue + beta * norm2loss
                elif Global_CDKT_metric == "JSD":
                    loss = lossTrue + beta * lossJSD

                self.optimizer.zero_grad()
                loss.backward()
                updated_model, _ = self.optimizer.step()

    def train(self):
        '''
        @Mao

        CDKT Server begin training process.
        '''
        print("CDKT begin training...")

        for glob_iter in range(self.num_glob_iters):
            # get users selected <= self.users
            self.selected_users = self.select_users(glob_iter, self.num_users)
            if(self.experiment):
                self.experiment.set_epoch(glob_iter + 1)
            print("-------------Round number: ", glob_iter, " -------------")
            # ============= Test each client =============
            tqdm.write(
                '============= Test Client Models - Specialization ============= ')
            stest_acu, strain_acc = self.evaluating_clients(
                glob_iter, mode="spe")
            self.cs_avg_data_test.append(stest_acu)
            self.cs_avg_data_train.append(strain_acc)
            tqdm.write(
                '============= Test Client Models - Generalization ============= ')
            gtest_acu, gtrain_acc = self.evaluating_clients(
                glob_iter, mode="gen")
            self.cg_avg_data_test.append(gtest_acu)
            # append an empty list
            self.cg_avg_data_train.append(gtrain_acc)

            tqdm.write('============= Test Global Models  ============= ')
            self.evaluating_global_CDKT(glob_iter)

            # NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                if(glob_iter == 0):
                    print("client train for the first round")
                    user.train(self.local_epochs)
                else:
                    print("client train distill...")
                    user.train_distill(self.local_epochs,
                                       self.model, glob_iter, alpha)

            if Ensemble == True:
                self.generalized_knowledge_ensemble(global_generalized_epochs)
            else:
                self.generalized_knowledge_construction(
                    global_generalized_epochs, glob_iter)

        self.save_results1()
        self.save_model()

    def save_results1(self):
        print("write result is empty: ", self.rs_train_acc)
        write_file(file_name=rs_file_path, root_test=self.rs_glob_acc, root_train=self.rs_train_acc,
                   cs_avg_data_test=self.cs_avg_data_test, cs_avg_data_train=self.cs_avg_data_train,
                   cg_avg_data_test=self.cg_avg_data_test, cg_avg_data_train=self.cg_avg_data_train,
                   cs_data_test=self.cs_data_test, cs_data_train=self.cs_data_train, cg_data_test=self.cg_data_test,
                   cg_data_train=self.cg_data_train, N_clients=[N_clients])
        plot_from_file()
