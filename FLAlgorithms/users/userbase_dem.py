import torch
import os
from torch.utils.data import DataLoader
import copy
from Setting import *
from utils.clustering.DTree import Node


class User(Node):
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, test_data, public_data, model, client_model, batch_size=0, learning_rate=0, beta=0, L_k=0, local_epochs=0, group=None):
        # from fedprox

        self.device = device
        self.model = copy.deepcopy(model)
        self.client_model = copy.deepcopy(client_model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.public_samples = len(public_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.L_k = L_k
        self.local_epochs = local_epochs
        self.group = group

        # DemLearn Client
        self._id = id
        self.group = group or []
        self._type = "Client"
        self.level = 0
        self.childs = None
        self.numb_clients = 1.0
        self.gmodel = self.model
        self.publicdatasetloader = None

        if(self.batch_size == 0):
            self.trainloader = DataLoader(
                train_data, self.train_samples, shuffle=True)
            self.testloader = DataLoader(
                test_data, self.test_samples, shuffle=True)
            # self.trainloaderpublic = DataLoader(train_public_data, self.train_public_samples, shuffle=True)
            self.publicloader = DataLoader(
                public_data, self.public_samples, shuffle=True)
        else:
            self.trainloader = DataLoader(
                train_data, self.batch_size, shuffle=True)
            # self.trainloaderpublic = DataLoader(train_public_data, self.batch_size, shuffle=True)
            self.publicloader = DataLoader(
                public_data, self.batch_size, shuffle=True)
            # self.publicdatasetloader = DataLoader(public_data, self.batch_size, shuffle=False) #no shuffle
            # list(self.publicdatasetloader)
            # for b, (x, y) in enumerate(self.publicdatasetloader):
            #     # self.publicdatasetlist+= (b,(x,y))
            #     print("A")

            # if(len(train_data) < 200):
            #     self.batch_size = int(len(test_data)/10)
            self.testloader = DataLoader(
                test_data, self.batch_size,  shuffle=True)

        self.testloaderfull = DataLoader(
            test_data, self.test_samples, shuffle=True)
        self.trainloaderfull = DataLoader(
            train_data, self.train_samples, shuffle=True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        # #self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        # self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))

    # def set_parameters(self, model):
    #     for old_param, new_param in zip(self.model.parameters(), model.parameters()):
    #         old_param.data = new_param.data.clone()
    #     self.gmodel = self.model
    def set_parameters(self, model=None):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
            # old_param.data = new_param.data
            # local_param.data = new_param.data
        self.gmodel = self.model

    def set_meta_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()
        self.gmodel = self.model

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        # print("server model is ",self.server_model)
        if Same_model:
            self.model.eval()
        else:
            self.client_model.eval()

        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            # output = self.model(x)
            if Same_model:
                output, _ = self.model(x)
            else:
                output, _ = self.client_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0], test_acc / y.shape[0]

    def test_gen(self, p_model):
        # self.model.eval()
        p_model.eval()
        test_acc = 0
        # self.update_parameters(p_model)

        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            # output = p_model(x)
            output, _ = p_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        # self.update_parameters(self.local_model)
        return test_acc, y.shape[0]
        #### The both implementation shows similar performance C-GEN for FedAvg and G-GEN in DemLearn but not C-GEN ??? ####
        # self.model.eval()
        # test_acc = 0
        # self.update_parameters(p_model.parameters())
        #
        # for x, y in self.testloaderfull:
        #     x, y = x.to(self.device), y.to(self.device)
        #     output = self.model(x)
        #     test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        # self.update_parameters(self.local_model)
        # return test_acc, y.shape[0]

    def train_error_and_loss(self):
        if Same_model:
            self.model.eval()
        else:
            self.client_model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            # output = self.model(x)
            if Same_model:
                output, _ = self.model(x)
            else:
                output, _ = self.client_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        return train_acc, loss.data.tolist(), self.train_samples

    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0], test_acc / y.shape[0]

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss.data.tolist(), self.train_samples

    def get_next_train_batch(self):
        if(self.batch_size == 0):
            for X, y in self.trainloaderfull:
                return (X.to(self.device), y.to(self.device))
        else:
            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_trainloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (X, y) = next(self.iter_trainloader)
            return (X.to(self.device), y.to(self.device))

    def get_next_test_batch(self):
        if(self.batch_size == 0):
            for X, y in self.testloaderfull:
                return (X.to(self.device), y.to(self.device))
        else:
            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_testloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_testloader = iter(self.testloader)
                (X, y) = next(self.iter_testloader)
            return (X.to(self.device), y.to(self.device))

    def get_alk(self, user_list, dataset, index):
        # temporary fix value of akl, all client has same value of akl
        # akl = 0.25 # can set any value but need to modify eta accordingly
        akl = 0.5
        #akl = 1
        return akl

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(
            model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
