#!/bin/bash

#三、训练模型

#3.1训练模型1
CUDA_VISIBLE_DEVICES=0 python train_LPRNet_xunfei6.py \
--pretrained_model './weights/Final_LPRNet_model.pth' \
--train_img_dirs '../xfdata/xunfei_train_val_ccpd2019_base_val_test.txt' \
--test_img_dirs '../xfdata/xunfei_carplate_train_val.txt' \
--save_folder '../user_data/tmp_data/xunfei_train_0818_3/' \
--learning_rate 0.0001 \
--lr_schedule 31 40 \
--max_epoch 50 > ./log/xunfei_train_0818_3.txt

#3.2训练模型2
CUDA_VISIBLE_DEVICES=0 python train_LPRNet_xunfei6.py \
--pretrained_model './weights/Final_LPRNet_model.pth' \
--train_img_dirs '../xfdata/xunfei_train_val_ccpd2019_base_val_test_crpd.txt' \
--test_img_dirs '../xfdata/xunfei_carplate_train_val.txt' \
--save_folder '../user_data/tmp_data/xunfei_train_0822_1/' \
--learning_rate 0.0001 \
--lr_schedule 31 40 \
--max_epoch 50 > ./log/xunfei_train_0822_1.txt

#3.3训练模型3
CUDA_VISIBLE_DEVICES=0 python train_LPRNet_xunfei6.py \
--pretrained_model './weights/Final_LPRNet_model.pth' \
--train_img_dirs '../xfdata/xunfei_train_ccpd2019_base_val_test_crpd.txt' \
--test_img_dirs '../xfdata/xunfei_carplate_train_val.txt' \
--save_folder '../user_data/tmp_data/xunfei_train_0818_4/' \
--learning_rate 0.0001 \
--lr_schedule 31 40 \
--max_epoch 50 > ./log/xunfei_train_0818_4.txt

#3.4训练模型4
CUDA_VISIBLE_DEVICES=0 python train_LPRNet_xunfei6.py \
--pretrained_model './weights/Final_LPRNet_model.pth' \
--train_img_dirs '../xfdata/xunfei_train_ccpd2019_base_val_test.txt' \
--test_img_dirs '../xfdata/xunfei_carplate_train_val.txt' \
--save_folder '../user_data/tmp_data/xunfei_train_0816_1/' \
--learning_rate 0.0001 \
--lr_schedule 31 40 \
--max_epoch 50 > ./log/xunfei_train_0816_1.txt


