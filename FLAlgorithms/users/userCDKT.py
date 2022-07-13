import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
# from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import DemProx_SGD
from FLAlgorithms.users.userbase_dem import User
from FLAlgorithms.trainmodel.models import *
# Implementation for clients
from utils.train_utils import KL_Loss, JSD

from Setting import *


class UserCDKT(User):
    def __init__(self, device, numeric_id, train_data, test_data, public_data, model, client_model, batch_size, learning_rate, beta,
                 local_epochs):
        super().__init__(device, numeric_id, train_data, test_data, public_data,  model[0], client_model, batch_size, learning_rate, beta,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # server and clients have the same model
        if Same_model:
            if Accelerated:
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=local_learning_rate)
            else:
                self.optimizer = DemProx_SGD(
                    self.model.parameters(), lr=local_learning_rate, mu=0)
        # different models
        else:
            self.optimizer = DemProx_SGD(
                self.client_model.parameters(), lr=local_learning_rate, mu=0)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        '''
        @Mao

        User train process with supervised learning.

        Args:
            `epochs`(int): num of local training epochs
        '''
        if Same_model:
            self.model.train()
        else:
            self.client_model.train()

        for _ in range(1, epochs + 1):

            for X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                if Same_model:
                    output, _ = self.model(X)
                else:
                    output, _ = self.client_model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

    def train_distill(self, epochs, global_model, glob_iter, alpha1=alpha):
        '''
        @Mao

        Training process with multi-losses.

        Args:
            `epochs`(int):
            `global_model`: server model
            `glob_iter`(int): global round index
            `alpha1`(float): alpha parameter
        '''
        gen_model = copy.deepcopy(global_model)
        if Same_model:
            self.model.train()
        else:
            self.client_model.train()

        batch_num = 0

        for _ in range(1, epochs + 1):  # local update

            if Same_model:
                self.model.train()
            else:
                self.client_model.train()

            private_data = enumerate(self.trainloader)
            # loop all of batch of public data
            public_data = self.publicdatasetloader[batch_num:]
            # print(type(private_data))
            # print(type(public_data))
            for [local_batch_idx, (X, y)], [batch_idx, (X_public, y_public)] in zip(private_data, public_data):
                print("local batch id: ", local_batch_idx,
                      " public batch ", batch_idx)
                X, y = X.to(self.device), y.to(self.device)
                X_public, y_public = X_public.to(
                    self.device), y_public.to(self.device)
                if batch_idx == 0:
                    print('local batch index = 0 ', X_public[0])

                if Same_model:
                    output_public, rep_output_public = self.model(X_public)
                    output, _ = self.model(X)
                else:
                    output_public, rep_output_public = self.client_model(
                        X_public)
                    output, _ = self.client_model(X)

                if(batch_idx < 1):
                    print('user output_public:', F.softmax(output_public))

                # gen model from server
                gen_output_public, rep_gen_output_public = gen_model(
                    X_public)

                print("gen public label previous", gen_output_public)
                if Tune_output:
                    y_onehot = F.one_hot(y_public, num_classes=NUMBER_LABEL)
                    gen_output_public = (
                        gen_output_public + y_onehot)/2.0  # lambda is 0.5

                print("gen public label after y hot: ", gen_output_public)

                lossTrue = self.loss(output, y)
                lossKD = lossJSD = norm2loss = 0
                if Full_model:
                    lossKD = self.criterion_KL(
                        output_public, gen_output_public)
                    norm2loss = torch.dist(
                        output_public, gen_output_public, p=2)
                    lossJSD = self.criterion_JSD(
                        output_public, gen_output_public)
                else:
                    lossKD = self.criterion_KL(
                        output_public, gen_output_public)
                    norm2loss = torch.dist(
                        output_public, gen_output_public, p=2)
                    lossJSD = self.criterion_JSD(
                        output_public, gen_output_public)
                    lossKD += self.criterion_KL(rep_output_public,
                                                rep_gen_output_public)
                    lossJSD += self.criterion_JSD(rep_output_public,
                                                  rep_gen_output_public)
                    norm2loss = norm2loss + \
                        torch.dist(rep_output_public,
                                   rep_gen_output_public, p=2)

                if Local_CDKT_metric == "KL":
                    loss = lossTrue + alpha1 * lossKD
                elif Local_CDKT_metric == "Norm2":
                    loss = lossTrue + alpha1 * norm2loss
                elif Local_CDKT_metric == "JSD":
                    loss = lossTrue + alpha1 * lossJSD

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                batch_num += 1

        # logits_dict = dict()
        # for batch_idx, (X_public,y_public) in enumerate(self.publicdatasetloader):
        #     X_public, y_public = X_public.to(self.device), y_public.to(self.device)
        #     logits_pub = self.model(X_public)
        #     logits_pub = logits_pub.cpu().detach().numpy()
        #     logits_dict[batch_idx] = logits_pub
        # print("all logit")
        # print(logits_dict)
        # print("finish all logits")

        # self.clone_model_paramenter(self.model.parameters(), self.local_model)

    def train_prox(self, epochs):
        '''
        @Mao

        I don't understand the function of this method

        Args:
            `epochs`(int): num of local epochs
        '''
        gen_model = copy.deepcopy(self.model)
        self.model.train()

        for epoch in range(1, epochs + 1):  # local update
            self.model.train()

            for X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)

                loss = self.loss(output, y)
                loss.backward()

                updated_model, _ = self.optimizer.step(
                    mu_t=1, gen_weights=(gen_model, 1.0))

        # update local model as local_weight_upated
        self.clone_model_paramenter(self.model.parameters(), self.local_model)

        # self.update_parameters(updated_model)
