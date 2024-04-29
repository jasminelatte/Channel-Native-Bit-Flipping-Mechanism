import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torchvision.models as models


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))


from cv.cnn import CNN_DropOut
from data_load import load_partition_data_federated_emnist

from FedAvgAPI.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from cv.resnet_gn import resnet18

from d2d_FL_train import FedAvgAPI
from FedAvgAPI.model.linear.lr import LogisticRegression
from FedAvgAPI.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from FedAvgAPI.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from FedAvgAPI.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
from FedAvgAPI.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet18_gn', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='fed_cifar100', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./data/fed_cifar100',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=10,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    parser.add_argument('--d2d_user_num', type=int, default=10, help='user_num of d2d decentralized')
    parser.add_argument('--label_divided_num', type=int, default=2, help='the number of labels divided to')
    parser.add_argument('--train_data_dir', type=str, default='fed_emnist_train.h5', help='train data')
    parser.add_argument('--Clipping_threshold', default=10, help='Clipping threshold: FMNIST-10; MNIST-10; CIFAR100-50')
    parser.add_argument('--privacy_budget', default=100000, help='eplison=100000,200000,300000,400000,500000')
    parser.add_argument('--different_model_bit_numbers', default=10000, help='different_model_bit_numbers')
    parser.add_argument('--Gaussian_indicator', default=0, help='Gaussian_indicator')
    parser.add_argument('--packet_length', default=1024, help='packet_length')
    parser.add_argument('--minimum_BSR', default=0.98, help='minimum_BSR')
    logging.info(parser)
    return parser


def load_data(args, dataset_name):
    # check if the centralized training is enabled


    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.d2d_user_num, args.data_dir, train_data_f=args.train_data_dir,label_divided_num=args.label_divided_num)

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    return model


def custom_model_trainer(args, model):
    if args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    else: # default model trainer is for classification problem
        return MyModelTrainerCLS(model)


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone-2'))
    args = parser.parse_args()
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ticks = time.gmtime()

    fh = logging.FileHandler('experiment_results/Noniid/%s%s%s%s_Gaussain_%s_minBSR_%s_epsilon_%s_L_%s_T_%s_PacketlengthL_%s.log'%(ticks.tm_mon,ticks.tm_mday,ticks.tm_hour,ticks.tm_min,args.Gaussian_indicator,args.minimum_BSR,args.privacy_budget,args.different_model_bit_numbers,args.comm_round,args.packet_length))
    logger.addHandler(fh)
    logger.info(ticks)
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)


    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    dataset = load_data(args, args.dataset)

    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)
    fedavgAPI.train()
