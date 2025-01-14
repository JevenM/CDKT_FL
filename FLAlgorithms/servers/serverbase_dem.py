import numpy as np
from tqdm import tqdm

from FLAlgorithms.servers.serverbase import Server
from utils.clustering.hierrachical_clustering import tree_construction, cal_linkage_matrix, weight_clustering, \
    gradient_clustering
import copy
from Setting import *


class Dem_Server(Server):
    def __init__(self, experiment, device, dataset, algorithm, model, client_model, batch_size, learning_rate, beta, L_k,
                 num_glob_iters, local_epochs, optimizer, num_users, times, args=None):
        super().__init__(experiment, device, dataset, algorithm, model, client_model, batch_size, learning_rate, beta, L_k, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # DEMLEARN PARAMS
        if(args is not None):
            if (args.mu > 0):
                args.algorithm += "_Prox"
                self.algorithm = args.algorithm

            self.num_rounds = num_glob_iters
            self.total_users = args.total_users
            self.N_clients = self.total_users
            self.selected_N_clients = int(self.num_users * self.total_users)
            self.args = args
            self.count_grs = np.zeros(K_Levels + 1)

        self.cg_avg_data_test = []  # avg generalization client accuracy test
        self.cg_avg_data_train = []  # avg generalization client accuracy train
        self.cs_avg_data_test = []  # avg specialization client test accuracy
        self.cs_avg_data_train = []  # avg specialization client train accuracy
        # self.gs_data_test = []  # specialization of group test accuracy
        # self.gs_data_test.append(np.zeros(K_Levels+1 ))
        # self.gg_data_test = [] # generalization of group test accuracy
        # self.gg_data_train = []  # generalization of group train accuracy
        # self.gs_data_train = []  # specialization of group train accuracy
        # self.gs_data_train.append(np.zeros(K_Levels + 1))

        self.cs_data_test = np.zeros(
            (self.num_rounds, self.selected_N_clients))
        self.cs_data_train = np.zeros(
            (self.num_rounds, self.selected_N_clients))
        self.cg_data_test = np.zeros(
            (self.num_rounds, self.selected_N_clients))
        self.cg_data_train = np.zeros(
            (self.num_rounds, self.selected_N_clients))
        # specialization of group train accuracy
        self.gs_level_train = np.zeros((K_Levels + 1, self.num_rounds, 2))
        # specialization of group test accuracy
        self.gs_level_test = np.zeros((K_Levels + 1, self.num_rounds, 2))
        # specialization of group k train accuracy
        self.gks_level_train = np.zeros((2, self.num_rounds))
        # specialization of group k test accuracy
        self.gks_level_test = np.zeros((2, self.num_rounds))
        # generalization of group train accuracy
        self.gg_level_train = np.zeros((K_Levels + 1, self.num_rounds, 2))
        # generalization of group test accuracy
        self.gg_level_test = np.zeros((K_Levels + 1, self.num_rounds, 2))
        # generalization of group k train accuracy
        self.gkg_level_train = np.zeros((2, self.num_rounds))
        # generalization of group k test accuracy
        self.gkg_level_test = np.zeros((2, self.num_rounds))
        self.dendo_data = []
        self.dendo_data_round = []
        self.time_complex = []

    def run_clustering_rep(self):
        p_list = []
        for c in self.users:
            # print("Weight:", w[1][0])
            # print("Bias:", w[1][1])
            # weight_idx = (K_Layer_idx[0] - 1)*2 #the layer for clustering algorithm e.g.: (3-1) *2 =4 (layer3, index 4, 5)
            # the layer for clustering algorithm e.g.: (3-1) *2 =4 (layer3, index 4, 5) (we can choose 6, 7 as layer 4 to cluster)
            weight_idx = (K_Layer_idx[1] - 1) * 2
            w_cnt = 0
            if (CLUSTER_METHOD == "weight"):
                c_w = []
                # for w in range(2):  #using the first layer to cluster only.
                for w in c.model.parameters():
                    # print("layer element idx:",w_cnt)
                    # using one layer for clustering (i.e., layer 3 => last layer of rep)
                    if(w_cnt > weight_idx) and (w_cnt < weight_idx+2):
                        c_w.append(w.data.flatten().cpu().numpy())
                    w_cnt += 1

                # print("Concatenation Len for clustering:",len(np.concatenate( c_w, axis=0)))
                p_list.append(np.concatenate(c_w, axis=0))
            else:
                c_g = []
                # for g in range(2): #using the first layer to cluster only.
                for g in c.model.parameters():
                    # using one layer for clustering (i.e., layer 3 => last layer of rep)
                    if(w_cnt > weight_idx) and (w_cnt < weight_idx+2):
                        c_g.append(g.data.flatten().cpu().numpy())
                    w_cnt += 1
                p_list.append(np.concatenate(c_g, axis=0))

        self.Weight_dimension = len(p_list[0])
        if (CLUSTER_METHOD == "weight"):
            return weight_clustering(p_list)
        else:
            return gradient_clustering(p_list)

    def run_clustering(self):
        p_list = []
        for c in self.users:
            # print("Weight:", w[1][0])
            # print("Bias:", w[1][1])
            if (CLUSTER_METHOD == "weight"):
                c_w = []
                w_cnt = 0
                # for w in range(2):  #using the first layer to cluster only.
                for w in c.model.parameters():
                    if(w_cnt < 2):
                        c_w.append(w.data.flatten().cpu().numpy())
                    w_cnt += 1
                # print("Concatenation Len for clustering:",len(np.concatenate( c_w, axis=0)))
                # print(f"Numb of parameters elements in NN {w_cnt}")
                concat_w = np.concatenate(c_w, axis=0)
                # print(len(concat_w))
                p_list.append(concat_w)
            else:
                c_g = []
                # for g in range(2): #using the first layer to cluster only.
                for g in c.model.parameters():
                    c_g.append(g.data.flatten().cpu().numpy())
                p_list.append(np.concatenate(c_g, axis=0))

        self.Weight_dimension = len(p_list[0])
        if (CLUSTER_METHOD == "weight"):
            return weight_clustering(p_list)
        else:
            return gradient_clustering(p_list)

    def initialize_model(self, user, init_model=None):
        assert (self.users is not None and len(self.users) > 0)
        if(init_model == None):
            user.set_parameters(self.model)
        else:
            user.set_parameters(init_model)

    # copy rep of left to right:
    def copy_rep(self, pre_w, new_w):
        w_cnt = 0
        # 3*2 = 6 (head start from element 6th)
        head_idx = (self.args.K_Layer_idx[0]-1) * 2
        for p_w, n_w in zip(pre_w.parameters(), new_w.parameters()):
            if (w_cnt < head_idx):  # load representation
                n_w.data = p_w.data
            else:  # do not need to load head layers
                break
            w_cnt += 1

    # copy rep of left to right:
    def copy_sub_rep(self, pre_w, new_w, rep_start, rep_end):
        w_cnt = 0
        for p_w, n_w in zip(pre_w.parameters(), new_w.parameters()):
            if (w_cnt >= rep_start) and (w_cnt <= rep_end):  # load representation
                n_w.data = p_w.data
            w_cnt += 1

    # # copy new presentation to replace old representation => fix head
    # def reverse_head(self, pre_w, new_w):
    #     w_cnt = 0
    #     head_idx = (self.args.K_Layer_idx[0]-1) * 2 ## 3*2 = 6 (head start from element 6th)
    #     for p_w, n_w in zip(pre_w.parameters(), new_w.parameters()):
    #         if (w_cnt < head_idx):  # load representation
    #             p_w.data = n_w.data
    #         else:  # do not need to load head layers
    #             break
    #         w_cnt += 1

    def update_generalized_model(self, round, node, mode="hard"):
        # print("Node id:", node._id, node._type)
        childs = node.childs
        if childs:
            # node.numb_clients = node.count_clients()
            node.in_clients = node.collect_clients()
            node.numb_clients = len(node.in_clients)
            # print(node.numb_clients)
            # print(self.Weight_dimension)

            rs_ws = copy.deepcopy(
                self.update_generalized_model(round, childs[0], mode))
            for rs_w in rs_ws.parameters():
                rs_w.data = rs_w.data * childs[0].numb_clients
            for child in childs[1:]:
                child_weights = self.update_generalized_model(
                    round, child, mode)
                for rs_w, child_w in zip(rs_ws.parameters(), child_weights.parameters()):
                    rs_w.data = rs_w.data + child.numb_clients * child_w.data  # weight

            if(mode == "hard"):
                for rs_w in rs_ws.parameters():
                    rs_w.data = rs_w.data / node.numb_clients
                    # if node.level == 1 and round<10:
                    #     rs_w.data *=1.1
            elif(mode == "soft"):
                for o_w, rs_w in zip(node.gmodel.parameters(), rs_ws.parameters()):
                    rs_w.data = (1 - self.args.gamma) * o_w.data + \
                        self.args.gamma * rs_w.data / node.numb_clients

            # node.gmodel = rs_ws
            node.set_node_parameters(rs_ws)
            return rs_ws

        elif (node._type.upper() == "CLIENT"):  # Client
            return node.model

    def update_generalized_model_re(self, round, node, mode="hard"):
        # print("Node id:", node._id, node._type)
        childs = node.childs
        if childs:
            # node.numb_clients = node.count_clients()
            node.in_clients = node.collect_clients()
            node.numb_clients = len(node.in_clients)
            # print(node.numb_clients)
            # print(self.Weight_dimension)
            rs_ws = copy.deepcopy(
                self.update_generalized_model_re(round, childs[0], mode))
            for rs_w in rs_ws.parameters():
                rs_w.data = rs_w.data * childs[0].numb_clients
            for child in childs[1:]:
                child_weights = self.update_generalized_model_re(
                    round, child, mode)
                for rs_w, child_w in zip(rs_ws.parameters(), child_weights.parameters()):
                    rs_w.data = rs_w.data + child.numb_clients * child_w.data  # weight

            if(node.parent == "Empty"):
                for rs_w in rs_ws.parameters():
                    rs_w.data = rs_w.data / node.numb_clients
            else:
                for p_w, rs_w in zip(node.parent.gmodel.parameters(), rs_ws.parameters()):
                    rs_w.data = (p_w.data + rs_w.data /
                                 node.numb_clients) * 0.5

            # node.gmodel = rs_ws
            node.set_node_parameters(rs_ws)
            return rs_ws

        elif (node._type.upper() == "CLIENT"):  # Client
            return node.model

    def update_generalized_model_downward(self, round, node, mode="hard"):
        # print("Node id:", node._id, node._type)
        mu_factor = 0.5
        childs = node.childs
        if childs:
            for child in childs[0:]:
                for c_w, n_w in zip(child.gmodel.parameters(), node.gmodel.parameters()):
                    if (mode == "hard"):
                        c_w.data = (1-mu_factor)*c_w.data + n_w.data*mu_factor
                    elif (mode == "soft"):
                        c_w.data = (1 - self.args.gamma) * c_w.data + \
                            self.args.gamma * (c_w.data + n_w.data)*0.5

                    # if (node.level == 2) and round<10:
                    #     c_w.data *=1.1
                self.update_generalized_model_downward(round, child, mode)

    def update_generalized_model_recursive2(self, round, node, mode="hard"):
        self.update_generalized_model(round, node, mode)  # bottom to top
        self.update_generalized_model_downward(
            round, node, mode)  # top to bottom

    def update_generalized_model_recursive1(self, round, node, mode="hard"):
        if(mode == "hard"):
            self.update_generalized_model_recursive2(round, node, mode)
        else:
            # bottom to top in recursive way
            self.update_generalized_model_re(round, node, mode)

    def get_hierrachical_params(self, client):
        gen_weights, nf = client.get_hierarchical_info1()
        # print("Normalized term:", nf)
        for w in gen_weights.parameters():
            w.data = w.data / nf

        # total_corrects, ns = self.gc_test(gen_weights)
        # print(f"G_GEN from download model: {total_corrects / ns}")
        return gen_weights.parameters()  # normalized version

    def get_hierrachical_gen_model(self, client):
        # gen_weights, nf = client.get_hierarchical_info1()
        gen_weights, nf = client.get_hierarchical_info()

        for w in gen_weights.parameters():
            w.data = w.data/nf
        return gen_weights, 1.0

        # return client.get_hierrachical_gen_model()
        # return client.get_hierarchical_info1()
        # return client.get_hierarchical_info()

    def get_hierrachical_representation(self, client):
        return client.get_hierarchical_rep().parameters()

        # model shape type: tuple ((weight, bias),(weight, bias))

    def hierrachical_clustering(self, i):
        # if(self.Hierrchical_Method == "Weight"):
        #     weights_matrix = self.create_matrix()
        #
        # else:
        #     gradient_matrix = self.create_matrix()
        #     # gradient_matrix = np.random.rand(N_clients, Weight_dimension)
        #     model = gradient_clustering(gradient_matrix)

        # start_cluster = timeit.timeit()
        if(self.args.algorithm == "DemLearn" or self.args.algorithm == "DemLearn_Prox"):
            model = self.run_clustering()
        elif(self.args.algorithm == "DemLearnRep" or self.args.algorithm == "DemLearnRep_Prox"):
            model = self.run_clustering_rep()
        # end_cluster = timeit.timeit()
        # total_time = start_cluster - end_cluster
        # self.time_complex.append(total_time)

        self.TreeRoot = tree_construction(model, self.users, round=i)
        # self.dendo_data.append([model, i])
        rs_linkage = cal_linkage_matrix(model)[1]
        self.dendo_data.append(rs_linkage)
        self.dendo_data_round.append(i)
        # self.dendo_data.append([rs_linkage, i])
        # plot_dendrogram(rs_linkage, round, self.alg)

        print("Number of agents in tree:", self.TreeRoot.count_clients())
        print("Number of agents in level K:", self.TreeRoot.childs[0].count_clients(),
              self.TreeRoot.childs[1].count_clients())
        # print("Number of agents Group 1 in level K-1:", root.childs[0].childs[0].count_clients(),
        #       root.childs[0].childs[1].count_clients())

    # mode spe: specialization, gen: generalization
    def g_train_error_and_loss(self, gr, mode="spe"):
        num_samples = []
        tot_correct = []
        losses = []

        # update parameter of group to tf.graph
        self.client_model.set_params(gr.gmodel)
        if (mode == "spe"):
            validating_clients = gr.in_clients
        elif (mode == "gen"):
            validating_clients = self.users

        for c in validating_clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        groups = [c.group for c in self.users]

        return ids, groups, num_samples, tot_correct, losses

    # C-GEN for Training data
    def gc_train_error_and_loss(self):
        num_samples = []
        tot_correct = []

        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

        return np.sum(tot_correct), np.sum(tot_correct), np.sum(num_samples)

    # mode spe: specialization, gen: generalization
    def c_train_error_and_loss(self, i, mode="spe"):
        num_samples = []
        tot_correct = []
        losses = []
        clients_acc = []

        for c in self.selected_users:
            if (mode == "spe"):
                ct, cl, ns = c.train_error_and_loss()
            elif (mode == "gen"):
                # Test client as testing group approach in gen mode
                ct, cl, ns = self.gc_train_error_and_loss()

            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
            clients_acc.append(ct / ns)

        # print("Training Acc Client:", clients_acc)
        # self.client_data_train[i][:] = clients_acc

        if (mode == "spe"):
            self.cs_data_train[i, :] = clients_acc
        elif (mode == "gen"):
            self.cg_data_train[i, :] = clients_acc

        ids = [c.id for c in self.users]
        groups = [c.group for c in self.users]

        return ids, groups, num_samples, tot_correct, losses

    def g_test(self, gr, mode="spe"):  # mode spe: specialization, gen: generalization
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []

        # print("Clients in group:",self.gr.in_clients)
        if (mode == "spe"):
            # Evaluate Group-SPE (clients belong to this group)
            validating_clients = gr.in_clients
        elif (mode == "gen"):
            # Evaluate Group-GEN (all clients data)
            validating_clients = self.users

        for c in validating_clients:
            # gr_test_model = copy.deepcopy(gr.gmodel)
            # gr_test_model= gr.gmodel.clone()
            # ct, ns = c.test_gen(gr_test_model)
            ct, ns = c.test_gen(gr.gmodel)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.users]
        groups = [c.group for c in self.users]

        return ids, groups, num_samples, tot_correct

    # C-GEN for Testing data
    def gc_test(self, c_model):
        num_samples = []
        tot_correct = []

        for c in self.users:
            ct, ns = c.test_gen(c_model)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

        return np.sum(tot_correct), np.sum(num_samples)

    def evaluating_global_CDKT(self, i):
        # evaluate global model on all client dataset
        ct, ns = self.gc_test(self.model)
        tqdm.write('At round {} global testing accuracy: {}'.format(i, ct / ns))
        self.rs_glob_acc.append(ct / ns)

    def c_test(self, i, mode="spe"):  # mode spe: specialization, gen: generalization
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        clients_acc = []

        for c in self.selected_users:
            # print("user id",c.id)
            if (mode == "spe"):
                ct, ns, _ = c.test()
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
            elif (mode == "gen"):
                # c_test_model = copy.deepcopy(c.model)
                # ct, ns = self.gc_test(c_test_model)  # Test client as testing group approach in gen mode
                if Same_model:
                    # Test client as testing group approach in gen mode
                    ct, ns = self.gc_test(c.model)
                else:
                    # Test client as testing group approach in gen mode
                    ct, ns = self.gc_test(c.client_model)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            clients_acc.append(ct / ns)

        # print("Testing Acc Client:", clients_acc )
        if (mode == "spe"):
            self.cs_data_test[i, :] = clients_acc
        elif (mode == "gen"):
            self.cg_data_test[i, :] = clients_acc
        # self.client_data_test.append(clients_acc)

        ids = [c.id for c in self.users]
        groups = [c.group for c in self.users]
        return ids, groups, num_samples, tot_correct

    # mode spe: specialization, gen: generalization
    def evaluating_clients(self, i, mode="spe"):
        # evaluate C-GEN and C-SPE on testing data
        stats = self.c_test(i, mode)
        if mode == "spe":
            stats_train = self.c_train_error_and_loss(
                i, mode)  # evaluate C-SPE on training data
        elif (mode == "gen"):
            stats_train = []  # no need to evaluate C-GEN on training data

        test_acr = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        tqdm.write('At round {} AvgC. testing accuracy: {}'.format(i, test_acr))
        if mode == "spe":
            train_acr = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
            tqdm.write(
                'At round {} AvgC. training accuracy: {}'.format(i, train_acr))
        elif (mode == "gen"):
            train_acr = []
        return test_acr, train_acr

    # mode spe: specialization, gen: generalization
    def evaluating_groups(self, gr, i, mode="spe"):
        if (gr.parent == "Empty"):
            self.test_accs = np.zeros(K_Levels + 1)
            self.train_accs = np.zeros(K_Levels + 1)
            self.count_grs = np.zeros(K_Levels + 1)

        stats = self.g_test(gr, mode)
        test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])

        if (mode == "spe"):  # Specialization results
            self.gs_level_train[gr.level - 1, i, 1] += gr.numb_clients
            self.gs_level_test[gr.level - 1, i,
                               0] += test_acc * gr.numb_clients
            self.gs_level_test[gr.level - 1, i, 1] += gr.numb_clients

            if (gr._id == self.TreeRoot.childs[0]._id):
                # self.gks_level_train[0,i] = train_acc
                self.gks_level_test[0, i] = test_acc
            elif (gr._id == self.TreeRoot.childs[1]._id):
                # self.gks_level_train[1,i] = train_acc
                self.gks_level_test[1, i] = test_acc

        elif (mode == "gen"):  # Generalization results
            # self.gg_level_train[gr.level - 1, i, 0] += train_acc * gr.numb_clients
            self.gg_level_train[gr.level - 1, i, 1] += gr.numb_clients
            self.gg_level_test[gr.level - 1, i,
                               0] += test_acc * gr.numb_clients
            self.gg_level_test[gr.level - 1, i, 1] += gr.numb_clients
            if (gr._id == self.TreeRoot.childs[0]._id):
                # self.gkg_level_train[0,i] = train_acc
                self.gkg_level_test[0, i] = test_acc
            elif (gr._id == self.TreeRoot.childs[1]._id):
                # self.gkg_level_train[1,i] = train_acc
                self.gkg_level_test[1, i] = test_acc

        if (gr.childs):
            for c in gr.childs:
                if (c._type.upper() == "GROUP"):
                    self.evaluating_groups(c, i, mode)
