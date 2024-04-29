import numpy as np
import copy
import torch
import random
import time
import logging
import struct
import concurrent.futures
import ctypes
import subprocess


# import thread
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

    # return string
    return str1
def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def add_noise_to_binary(number,error_probability):
    binary_str = binary(number)
    error=np.random.binomial(1,1-error_probability,len(binary_str))
    binary_array = list(binary_str)
    binary_array = [int(i) for i in binary_array]
    for i in range(0,len(binary_str)):
        if i<9:
            binary_array[i] = str(binary_array[i])
        else:
            if error[i]==0:
                binary_array[i]=str(1-binary_array[i])
            else:
                binary_array[i]=str(binary_array[i])
    return listToString(binary_array)
def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def bit_noise_add(communication_BSR,artificial_BSR,clipping_threshold,w):#READ THE VALUE OF EACH TORCH.TENSOR W AND THEN MODIFY all of them.

    wglobal=copy.deepcopy(w)
    for wkey in wglobal.keys():
        original_shape = wglobal[wkey].size()


        flat_tensor = wglobal[wkey].view(-1)
        # max_model_parameter=torch.max(wglobal[wkey])
        # binary_representation = bin(max_model_parameter)
        # print('binary_representation')
        # print(wkey)
        # print(original_shape)
        # print(max_model_parameter)
        # print(flat_tensor)
        original_FlatVector = flat_tensor.tolist() #一个随机向量
        FlatVector = [x + 3*clipping_threshold for x in original_FlatVector]
        # print(FlatVector)

        lib = ctypes.CDLL('./noise_add.so')
        lib.parallelProcess.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,ctypes.c_float]
        lib.parallelProcess.restype = ctypes.POINTER(ctypes.c_float)

        input_array = (ctypes.c_float * len(FlatVector))(*FlatVector)
        # print(input_array)
        input_it_error_rate=(ctypes.c_float)(communication_BSR)
        result=lib.parallelProcess(input_array,len(FlatVector),input_it_error_rate)
        if artificial_BSR<=1:
            input_it_error_rate2 = (ctypes.c_float)(artificial_BSR)
            result2 = lib.parallelProcess(result, len(FlatVector), input_it_error_rate2)
        # print(result)
            mapped_noised_vector=list(result2[0:len(FlatVector)])
        else:
            mapped_noised_vector = list(result[0:len(FlatVector)])
        noised_vector=[x - 3 * clipping_threshold for x in mapped_noised_vector]
        # lib.free_float_vector(result)
        flat_tensor2 = torch.tensor(noised_vector, dtype=torch.float32)
        modified_tensor = flat_tensor2.view(original_shape)
        wglobal[wkey]=copy.deepcopy(modified_tensor)
        # end = time.time()
        # logging.info("modified time = " + str(end - start))
    return wglobal

def process_item(item, error_rate):
    current_value = item.item()
    modified_value = add_noise_to_binary(current_value, error_rate)
    return bin_to_float(modified_value)
def parallel_process_chunk(chunk, error_rate):
    # print('1')
    # print(len(chunk))
    start = time.time()
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     modified_values = list(executor.map(lambda item: process_item(item, error_rate), chunk))

    modified_values = list(process_item(item, error_rate) for item in chunk)
    end = time.time()
    logging.info("parallel_process_chunk = " + str(end - start))
    return modified_values

def bit_noise_add_new(error_rate, w):#READ THE VALUE OF EACH TORCH.TENSOR W AND THEN MODIFY all of them.

    wglobal=copy.deepcopy(w)
    for wkey in wglobal.keys():
        logging.info(wkey)
        start = time.time()
        original_shape = wglobal[wkey].size()
        end = time.time()
        logging.info("original_shape = " + str(end - start))


        start = time.time()
        flat_tensor = wglobal[wkey].view(-1)
        end = time.time()
        logging.info("flaten_time = " + str(end - start))

        start = time.time()
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     modified_values = list(executor.map(lambda item: process_item(item, error_rate), flat_tensor))
        chunk_size = 100000  # Adjust the chunk size based on your data size
        chunks = [flat_tensor[i:i + chunk_size] for i in range(0, flat_tensor.size(0), chunk_size)]
        end = time.time()
        logging.info("chunks time= " + str(end - start))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            modified_chunks = list(executor.map(lambda chunk: parallel_process_chunk(chunk, error_rate), chunks))
        # modified_chunks = list(parallel_process_chunk(chunk, error_rate) for chunk in chunks)
        end = time.time()
        logging.info("add noise time= " + str(end - start))

        start = time.time()
        flat_tensor[:] = torch.cat([torch.tensor(chunk) for chunk in modified_chunks])
        # flat_tensor[:] = torch.tensor(modified_values)
        end = time.time()
        logging.info("flatten_time = " + str(end - start))
        #
        #
        # flat_tensor[:] = torch.cat([torch.tensor(chunk) for chunk in modified_chunks])



        start = time.time()
        modified_tensor = flat_tensor.view(original_shape)
        wglobal[wkey]=copy.deepcopy(modified_tensor)
        end = time.time()
        logging.info("modified time= " + str(end - start))
        # end = time.time()
        # logging.info("modified time = " + str(end - start))
    return wglobal

def users_sampling(args, w, chosenUsers):
    if args.num_chosenUsers < args.num_users:
        w_locals = []
        for i in range(len(chosenUsers)):
            w_locals.append(w[chosenUsers[i]])
    else:
        w_locals = copy.deepcopy(w)
    return w_locals
def Gaussian_noise_add(args,Gaussian_variance,w):
    w_noise = copy.deepcopy(w)

    for i in w_noise.keys():
        noise = np.random.normal(0, Gaussian_variance, w[i].size())
        # if args.gpu != -1:
        noise = torch.from_numpy(noise).float()
        # else:
        #     noise = torch.from_numpy(noise).float()
        w_noise[i] = w_noise[i] + noise
    return w_noise