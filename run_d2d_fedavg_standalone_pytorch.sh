#!/usr/bin/env bash

GPU=$1

BATCH_SIZE=$2

DATASET=$3

DATA_PATH=$4

MODEL=$5

ROUND=$6

EPOCH=$7

LR=$8

OPT=${9}

d2d_user_num=${10}

train_data=${11}

Clipping_threshold=${12}

Privacy_budget=${13}

Different_model_bit_numbers=${14}

Gaussian_indicator=${15}

packet_length=${16}

minimum_BSR=${17}
python ./mainn.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR \
--d2d_user_num $d2d_user_num \
--train_data_dir $train_data \
--Clipping_threshold $Clipping_threshold \
--privacy_budget $Privacy_budget \
--different_model_bit_numbers $Different_model_bit_numbers \
--Gaussian_indicator $Gaussian_indicator \
--packet_length $packet_length \
--minimum_BSR $minimum_BSR