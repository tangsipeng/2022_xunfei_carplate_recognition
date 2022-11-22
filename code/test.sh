#!/bin/bash
#二、结果复现
#2.1、利用已经训练好的模型复现结果:已经训练好的模型存放于 prject/user_data/model_data 目录下，总共有8个模型
#推理1
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0818_3/LPRNet__iteration_108000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0818_3/STNet__iteration_108000.pth' \
--save './xunfeisave/submit_0818_3_1a.csv'
#推理2
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0822_1/LPRNet__iteration_110000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0822_1/STNet__iteration_110000.pth' \
--save './xunfeisave/submit_0822_2a.csv'
#推理3
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0818_4/LPRNet__iteration_112000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0818_4/STNet__iteration_112000.pth' \
--save './xunfeisave/submit_0818_4a.csv'
#推理4
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0816_1/LPRNet__iteration_108000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0816_1/STNet__iteration_108000.pth' \
--save './xunfeisave/submit_0816_1a.csv'
#2.2、模型集成，生成 prediction_result目录下result.csv 文件
python mix_data_submit.py
