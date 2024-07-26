import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, ConnectionPatch
import math
# from lwc_d2dFL.topology_update import TopologyManager

import copy
import logging
import random
import time
from scipy.stats import bernoulli
import torch
from NoiseAdd import bit_noise_add
from NoiseAdd import Gaussian_noise_add
from NoiseAdd import bit_noise_add_gaussian



from FedAvgAPI.client import Client
DEFAULT_TRAIN_CLIENTS_NUM = 1000
DEFAULT_TEST_CLIENTS_NUM = 1000

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        # self.topology = topology
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        # self.client_ids = train_client_ids
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.d2d_user_num):
            print(client_idx)
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _BER_calculate(self):
        # numerator = math.exp((math.log(float(self.args.privacy_tail_bound)) + float(self.args.privacy_budget)) / (2 * float(self.args.comm_round) * float(self.args.different_model_bit_numbers)))
        denominator = 1 + math.exp((math.log(float(self.args.privacy_tail_bound)) + float(self.args.privacy_budget)) / (2 * float(self.args.comm_round) * float(self.args.different_model_bit_numbers)))
        return 1 / denominator

    def train(self):
        # logging.info(self.model_trainer)
        w_global = self.model_trainer.get_model_params()
        if self.args.dataset=='mnist':
            for k in w_global:
                w_global[k]=0.05*torch.ones_like(w_global[k])
                w_global[k]=torch.bernoulli(w_global[k])-w_global[k]


        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []



            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            if self.args.dataset=='mnist':
                client_indexes = self._client_sampling(6, DEFAULT_TRAIN_CLIENTS_NUM, self.args.d2d_user_num)
            elif self.args.dataset=='fed_cifar100':
                client_indexes = self._client_sampling(6, 10, self.args.d2d_user_num)
            else:
                client_indexes = self._client_sampling(6, DEFAULT_TRAIN_CLIENTS_NUM, self.args.d2d_user_num)
            logging.info("client_indexes = " + str(client_indexes))
            communication_PERs=[]
            for idx, client in enumerate(self.client_list):#user iteration change
                logging.info("client_indexes = " + str(idx))
                client_idx = client_indexes[idx]
               
                client.update_local_dataset(idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                start = time.time()
                w = client.train( copy.deepcopy(w_global))
                end = time.time()
                noised_w=None
                if self.args.Gaussian_indicator=='0': # adaptive  artificial bit flipping

                    BER = self._BER_calculate()

                    communication_BER = 1 - random.uniform(float(self.args.minimum_BSR), 1)

                    artificial_BER = (BER - communication_BER) / (1 - 2 * communication_BER)
                    print("Artificial_BER: " + str(artificial_BER))
                    if artificial_BER > 0:
                        logging.info(
                            "communication_BER: " + str(communication_BER) + ", artificial_BER: " + str(artificial_BER))
                        # 这里生成随机数，作为通信ber，然后给出计算人工ber的计算公式
                        logging.info('BER = ' + str(BER))
                    else:
                        logging.info(
                            "communication_BER: " + str(communication_BER) + ", artificial_BER: " + str(0))
                        # 这里生成随机数，作为通信ber，然后给出计算人工ber的计算公式
                        logging.info('BER = ' + str(communication_BER))
                        artificial_BER=0

                    noised_w = bit_noise_add(1 - communication_BER, 1 - artificial_BER,
                                             float(self.args.Clipping_threshold), w)
                elif self.args.Gaussian_indicator=='2': #Gaussain Mechanism and aggregate with all received model.
                    Gaussian_variance = self.Gaussian_BER_variance()

                    communication_PER = 1 - random.uniform(float(self.args.minimum_BSR), 1)**(int(self.args.packet_length))

                    logging.info("communication_PER: " + str(communication_PER)+ "packet_length: " + str(self.args.packet_length)+ ", Gaussian_variance: " + str(
                        Gaussian_variance))

                    noised_w = Gaussian_noise_add(self.args, Gaussian_variance, w)
                    communication_PERs.append(communication_PER)

                elif self.args.Gaussian_indicator=='1': #communication noise: Bit flipping; Artifical noise: Gaussain Noise
                #add noise here
                    Gaussian_variance=self.Gaussian_BER_variance()

                    communication_BER= 1-random.uniform(float(self.args.minimum_BSR), 1)


                    logging.info("communication_BER: "+str(communication_BER)+", Gaussian_variance: "+str(Gaussian_variance))


                    noised_Gaussian_w=Gaussian_noise_add(self.args,Gaussian_variance,w)
                    noised_w=bit_noise_add_gaussian(1-communication_BER,1,float(self.args.Clipping_threshold),noised_Gaussian_w)
                elif self.args.Gaussian_indicator=='3': # bit flipping

                    Artificial_BER = self._BER_calculate()

                    communication_BER = 1 - random.uniform(float(self.args.minimum_BSR), 1)

                    Joint_BER = Artificial_BER+communication_BER-2*communication_BER*Artificial_BER
                    # print("Artificial_BER: " + str(Artificial_BER))
                    # if artificial_BER > 0:
                    logging.info(
                        "communication_BER: " + str(communication_BER) + ", artificial_BER: " + str(Artificial_BER))
                    # 这里生成随机数，作为通信ber，然后给出计算人工ber的计算公式
                    logging.info('Joint_BER = ' + str(Joint_BER))
                    # else:
                    #     logging.info(
                    #         "communication_BER: " + str(communication_BER) + ", artificial_BER: " + str(0))
                    #     # 这里生成随机数，作为通信ber，然后给出计算人工ber的计算公式
                    #     logging.info('BER = ' + str(communication_BER))

                    noised_w = bit_noise_add(1 - communication_BER, 1 - Artificial_BER,
                                             float(self.args.Clipping_threshold), w)
                logging.info("training time = " + str(end - start))
                w_locals.append((client.get_sample_number(), copy.deepcopy(noised_w)))


            if self.args.Gaussian_indicator == '2':
                w_global = self._centralized_aggregation_PER_replace_with_averaged_model(w_locals,communication_PERs)
            else:
                w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)


            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx,client_indexes,w_global)
                # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx,client_indexes,w_global)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num  = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        logging.info("len(w_locals) " + str(len(w_locals)))
        # U = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num
        # for idx in range(len(w_locals)):
        #     (sample_num, averaged_params) = w_locals[idx]
        #     ww = sample_num/training_num
        #     U += ww*self.it_list[idx]
        # logging.info("U " + str(U))
        # \
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params


    def _local_test_on_one_client(self, round_idx, client_id):
        logging.info("################local_test_on_%s_client : %s th user"%(client_id, round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        client = self.client_list[client_id]
        # logging.info("client：%s"%client.client_idx)

        client.update_local_dataset(client_id, self.train_data_local_dict[client_id],
                                    self.test_data_local_dict[client_id],
                                    self.train_data_local_num_dict[client_id])
        # train data
        train_local_metrics = client.local_test(False)
        train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
        train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
        train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

        # test data
        test_local_metrics = client.local_test(True)
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

        """
        Note: CI environment is CPU-based computing. 
        The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        """


        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        # wandb.log({"Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_all_clients(self, round_idx,client_indexes,w_global):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))


        client = self.client_list[0]
        # logging.info("client：%s"%client.client_idx)

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        # self.model_trainer.set_model_params(w_locals[0])
        for client_idx in range(self.args.d2d_user_num):

            idxtest = client_indexes[client_idx]
            # logging.info("local_test_on_all_clients: test data of:{}".format(idxtest))
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[idxtest],
                                        self.test_data_local_dict[idxtest],
                                        self.train_data_local_num_dict[idxtest])
            # train data

            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        # wandb.log({"Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
    def _centralized_aggregation_PER_replace_with_averaged_model(self, w_locals,communication_PERs):
        training_num = 0
        logging.info("len(w_locals) " + str(len(w_locals)))
        # U = 0
        (sample_num, averaged_params) = w_locals[0]
        # original_shape = averaged_params.shape

        flattened = torch.cat([averaged_params[key].flatten() for key in averaged_params])
        split_tensor = torch.split(flattened, int(int(self.args.packet_length )/ 24))
        # split_shape=split_tensor.shape

        indicators_flattened=[]
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num
            indicator_idx_flatten=[]
            random_array = np.random.rand(len(split_tensor))
            binary_array = (random_array > communication_PERs[idx]).astype(int)
            for i, split in enumerate(split_tensor):
                indicator_idx_i=torch.ones(split_tensor[i].shape)



                # print(i)


                indicator_idx_flatten.append(binary_array[i]*sample_num*indicator_idx_i)
            # if idx==0:
            indicators_flattened.append(torch.cat(indicator_idx_flatten))
            # else:
            #     indicators_flattened=torch.stack((indicators_flattened,torch.cat(indicator_idx_flatten)),0)
        sum_indicators_flattened = torch.sum(torch.stack(indicators_flattened), dim=0)
        for idx in range(len(w_locals)):
            indicators_flattened[idx]=torch.where(sum_indicators_flattened != 0, indicators_flattened[idx] / sum_indicators_flattened, torch.zeros_like(indicators_flattened[idx]))
        # for idx in range(len(w_locals)):
        #     (sample_num, averaged_params) = w_locals[idx]
        #     ww = sample_num/training_num
        #     U += ww*self.it_list[idx]
        # logging.info("U " + str(U))
        # \

        averaged_params_flattened=None
        for i in range(0, len(w_locals)):

            local_sample_number, local_model_params = w_locals[i]

            local_model_params_flattened= torch.cat([local_model_params[key].flatten() for key in local_model_params])
            # print(indicators_flattened[i])
            if i == 0:
                averaged_params_flattened = torch.mul(local_model_params_flattened,indicators_flattened[i])
            else:
                averaged_params_flattened +=torch.mul(local_model_params_flattened,indicators_flattened[i])

        reconstructed_state_dict = {}
        index = 0
        for key in averaged_params:
            tensor = averaged_params[key]
            flattened_size = tensor.numel()
            # 根据张量的形状重新构建张量
            reconstructed_tensor = averaged_params_flattened[index: index + flattened_size].reshape(tensor.shape)
            # 将重建的张量放回新的OrderedDict
            reconstructed_state_dict[key] = reconstructed_tensor
            index += flattened_size

        return reconstructed_state_dict

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Pre": test_pre, "round": round_idx})
            # wandb.log({"Test/Rec": test_rec, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!"%self.args.dataset)

        logging.info(stats)

    def Gaussian_BER_variance(self):
        delta_s = 1.6 * 10 ** (-4)

        noise_scale = delta_s * float(self.args.comm_round)* np.sqrt(2  * np.log(1.25 / 0.25)) / (float(self.args.privacy_budget)*0.02/float(self.args.different_model_bit_numbers))

        return noise_scale