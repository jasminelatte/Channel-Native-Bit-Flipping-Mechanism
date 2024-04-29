import logging
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLIENTS_NUM = 3395
DEFAULT_TEST_CLIENTS_NUM = 3400
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = 'fed_emnist_train.h5'
DEFAULT_TEST_FILE = 'fed_emnist_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMGAE = 'pixels'
_LABEL = 'label'


def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None, train_data_f=DEFAULT_TRAIN_FILE):
    train_h5 = h5py.File(os.path.join(data_dir, train_data_f), 'r')
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), 'r')


    # load data
    if client_idx is None:
        # get ids of all clients
        train_ids = client_ids_train
        test_ids = client_ids_test
    else:
        # get ids of single client
        # logging.info(client_ids_train)
        train_ids = [client_ids_train[client_idx]]
        test_ids = [client_ids_test[client_idx]]

    # load data in numpy format from h5 file
    # logging.info(train_ids)
    train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in train_ids])
    train_y = np.vstack([train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in train_ids]).squeeze()
    test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in test_ids])
    test_y = np.vstack([test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in test_ids]).squeeze()

    # dataloader
    train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=True,
                               drop_last=False)
    # logging.info("client_idx = "+str(client_idx))
    # logging.info("train_ds = " + str(len(train_ds))+";train_dl = " + str(len(train_dl)))
    test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    test_dl = data.DataLoader(dataset=test_ds,
                              batch_size=test_bs,
                              shuffle=True,
                              drop_last=False)
    # logging.info("test_ds = " + str(len(test_ds)) + ";test_dl = " + str(len(test_dl)))

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl

def get_dataloader_noniid(dataset, data_dir, train_bs, test_bs, client_idx=None,train_data_f=DEFAULT_TRAIN_FILE):
    train_h5 = h5py.File(os.path.join(data_dir, train_data_f), 'r')
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), 'r')


    # load data
    if client_idx is None:
        # get ids of all clients
        train_ids = client_ids_train
        test_ids = client_ids_test
    else:
        # get ids of single client
        train_ids = [client_ids_train[client_idx]]
        test_ids = [client_ids_test[client_idx]]

    # load data in numpy format from h5 file
    train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in train_ids])
    train_y = np.vstack([train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in train_ids]).squeeze()
    test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in test_ids])
    test_y = np.vstack([test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in test_ids]).squeeze()

    # dataloader
    train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=True,
                               drop_last=False)
    # logging.info("client_idx = "+str(client_idx))
    # logging.info("train_ds = " + str(len(train_ds))+";train_dl = " + str(len(train_dl)))
    test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    test_dl = data.DataLoader(dataset=test_ds,
                              batch_size=test_bs,
                              shuffle=True,
                              drop_last=False)
    # logging.info("test_ds = " + str(len(test_ds)) + ";test_dl = " + str(len(test_dl)))

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl
def load_partition_data_distributed_federated_emnist(process_id, dataset, data_dir,
                                                     batch_size=DEFAULT_BATCH_SIZE, train_data_f=DEFAULT_TRAIN_FILE):
    if process_id == 0:
        # get global dataset
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1,train_data_f=train_data_f)
        train_data_num = len(train_data_global)
        # logging.info("train_dl_global number = " + str(train_data_num))
        # logging.info("test_dl_global number = " + str(test_data_num))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        train_file_path = os.path.join(data_dir, train_data_f)
        test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
        with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
            global client_ids_train, client_ids_test
            client_ids_train = list(train_h5[_EXAMPLE].keys())
            client_ids_test = list(test_h5[_EXAMPLE].keys())
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1,train_data_f=train_data_f)
        train_data_num = local_data_num = len(train_data_local)
        train_data_global = None
        test_data_global = None

    # class number
    train_file_path = os.path.join(data_dir, train_data_f)
    with h5py.File(train_file_path, 'r') as train_h5:
        class_num = len(np.unique(
            [train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0] for idx in range(DEFAULT_TRAIN_CLIENTS_NUM)]))
        logging.info("class_num = %d" % class_num)

    return DEFAULT_TRAIN_CLIENTS_NUM, train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_federated_emnist(dataset, user_num, data_dir, batch_size=DEFAULT_BATCH_SIZE, train_data_f=DEFAULT_TRAIN_FILE,label_divided_num=10):
    # client ids
    logging.info("load_partition_data_federated_emnist")
    # logging.info("load_partition_data_federated_emnist")
    train_file_path = os.path.join(data_dir, train_data_f)
    test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())
    # logging.info(client_ids_train)
    # local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    # step=DEFAULT_TRAIN_CLIENTS_NUM//user_num
    # train_client_ids=[]
    # for idx in range(user_num):
    #
    #     train_client_ids.append(np.random.randint(1, step)+idx*step)
    # logging.info(train_client_ids)

    # test_client_ids =  np.random.randint(1, DEFAULT_TRAIN_CLIENTS_NUM, user_num)
    if label_divided_num==10:
        default_train_num=10
    else:
        default_train_num=DEFAULT_TRAIN_CLIENTS_NUM
    for client_idx in range(default_train_num):
        # logging.info(client_idx)
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, client_idx,train_data_f=train_data_f)
        local_data_num = len(train_data_local) + len(test_data_local)
        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_train_number = %d" % (client_idx, len(train_data_local)))
        # logging.info("client_idx = %d, local_test_number = %d" % (client_idx, len(test_data_local)))
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #   client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # global dataset
    train_data_global = data.DataLoader(
        data.ConcatDataset(
            list(dl.dataset for dl in list(train_data_local_dict.values()))
        ),
        batch_size=batch_size, shuffle=True)
    train_data_num = len(train_data_global.dataset)

    test_data_global = data.DataLoader(
        data.ConcatDataset(
            list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
        ),
        batch_size=batch_size, shuffle=True)
    test_data_num = len(test_data_global.dataset)

    # class number
    train_file_path = os.path.join(data_dir, train_data_f)
    with h5py.File(train_file_path, 'r') as train_h5:
        class_num = len(np.unique(
            [train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0] for idx in range(default_train_num)]))
        logging.info("class_num = %d" % class_num)

    return user_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

