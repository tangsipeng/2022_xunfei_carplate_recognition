## 机动车车牌识别挑战赛 TOP1 方案 
### 测试数据可以通过 https://challenge.xfyun.cn/topic/info?type=license-plate-recognition&option=stsj 下载后放到 data/test目录下  
### 训练数据除了官方提供的train之外还需要自己下载ccpd2019和crpd两个数据集  

## 零、整体思路介绍  
### 0.1、数据分析：增加训练数据ccpd2019和crpd两个数据集，提升精度  
### 0.2、模型：输入尺寸(94，24)，STN对输入图片进行矫正，LPRnet输出车牌识别结果，保证推理速度  
### 0.3、数据增强：随机亮度和对比度，提升精度，单模型精度在0.986左右  
### 0.4、模型集成：根据输出置信度将4个模型集成（对应不同数据集组合方式），提升精度  
### 0.5、提交结果打分：预计在0.9918~0.992之间  
### 0.6、结果分析：由于分布接近，ccpd2019数据集加入可以大幅提升结果精度，而crpd数据加入对非皖开头车牌有帮助  


## 一、数据准备：运行train.sh test.sh test_2.sh之前必须先运行本节以下内容的命令  
### 1.1、解压数据  
cd project/xfdata  
unzip 机动车车牌识别挑战赛公开数据.zip  
mv 机动车车牌识别挑战赛公开数据/train ./  
mv 机动车车牌识别挑战赛公开数据/test ./  
cd ../code  

### 1.2、构造测试集序列文本、训练集序列文本  
python create_xunfei_test_text.py
python create_xunfei_train_text.py  
python create_xunfei_ccpd2019base_val_test.py  
python make_crpd_datasets.py  

### 1.3、合成训练数据序列 训练集-验证集-ccpd2019  
cat ../xfdata/xunfei_carplate_train_val.txt > ../xfdata/xunfei_train_val_ccpd2019_base_val_test.txt  
cat ../xfdata/xunfei_carplate_train.txt >> ../xfdata/xunfei_train_val_ccpd2019_base_val_test.txt  
cat ../xfdata/xunfei_ccpd2019_base_val_test.txt >> ../xfdata/xunfei_train_val_ccpd2019_base_val_test.txt  
### 验证数据量  
cat ../xfdata/xunfei_train_val_ccpd2019_base_val_test.txt | wc -l  
#363005  
 
### 1.4、合成训练数据序列 训练集-验证集-ccpd2019-crpd  
cat ../xfdata/xunfei_train_val_ccpd2019_base_val_test.txt > ../xfdata/xunfei_train_val_ccpd2019_base_val_test_crpd.txt  
cat ../xfdata/CRPD_cut.txt >> ../xfdata/xunfei_train_val_ccpd2019_base_val_test_crpd.txt  
#验证数据量  
cat ../xfdata/xunfei_train_val_ccpd2019_base_val_test_crpd.txt | wc -l  
#383541  

### 1.5、合成训练数据序列 训练集-ccpd2019-crpd  
cat ../xfdata/xunfei_carplate_train.txt > ../xfdata/xunfei_train_ccpd2019_base_val_test_crpd.txt  
cat ../xfdata/xunfei_ccpd2019_base_val_test.txt >> ../xfdata/xunfei_train_ccpd2019_base_val_test_crpd.txt  
cat ../xfdata/CRPD_cut.txt >> ../xfdata/xunfei_train_ccpd2019_base_val_test_crpd.txt  
#验证数据量  
cat ../xfdata/xunfei_train_ccpd2019_base_val_test_crpd.txt | wc -l  
#379335  

### 1.6、合成训练数据序列 训练集-ccpd2019  
cat ../xfdata/xunfei_carplate_train.txt > ../xfdata/xunfei_train_ccpd2019_base_val_test.txt  
cat ../xfdata/xunfei_ccpd2019_base_val_test.txt >> ../xfdata/xunfei_train_ccpd2019_base_val_test.txt  
#验证数据量  
cat ../xfdata/xunfei_train_ccpd2019_base_val_test.txt | wc -l  
#358799  


## 二、结果复现: 对应在code目录下运行 bash test.sh 则不必运行本节以下命令  
### 2.1、利用已经训练好的模型复现结果:已经训练好的模型存放于 prject/user_data/model_data 目录下，总共有8个模型  
### 推理1  
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0818_3/LPRNet__iteration_108000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0818_3/STNet__iteration_108000.pth' \
--save './xunfeisave/submit_0818_3_1a.csv'  
### 推理2  
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0822_1/LPRNet__iteration_110000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0822_1/STNet__iteration_110000.pth' \
--save './xunfeisave/submit_0822_2a.csv'  
### 推理3  
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0818_4/LPRNet__iteration_112000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0818_4/STNet__iteration_112000.pth' \
--save './xunfeisave/submit_0818_4a.csv'  
### 推理4  
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0816_1/LPRNet__iteration_108000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0816_1/STNet__iteration_108000.pth' \
--save './xunfeisave/submit_0816_1a.csv'  
### 2.2、模型集成，生成 prediction_result目录下result.csv 文件  
python mix_data4.py  


## 三、训练模型：对应在code目录下运行 bash train.sh > ./log/train.txt 2>&1 & 则不必运行本节以下命令  
### 训练占用显存6104M，V100训练单模大约需要4小时，总共需要16小时  
### 3.1训练模型1  
CUDA_VISIBLE_DEVICES=0 python train_LPRNet_xunfei6.py \
--pretrained_model './weights/Final_LPRNet_model.pth' \
--train_img_dirs '../xfdata/xunfei_train_val_ccpd2019_base_val_test.txt' \
--test_img_dirs '../xfdata/xunfei_carplate_train_val.txt' \
--save_folder '../user_data/tmp_data/xunfei_train_0818_3/' \
--learning_rate 0.0001 \
--lr_schedule 31 40 \
--max_epoch 50 > ./log/xunfei_train_0818_3.txt  

### 3.2训练模型2  
CUDA_VISIBLE_DEVICES=0 python train_LPRNet_xunfei6.py \
--pretrained_model './weights/Final_LPRNet_model.pth' \
--train_img_dirs '../xfdata/xunfei_train_val_ccpd2019_base_val_test_crpd.txt' \
--test_img_dirs '../xfdata/xunfei_carplate_train_val.txt' \
--save_folder '../user_data/tmp_data/xunfei_train_0822_1/' \
--learning_rate 0.0001 \
--lr_schedule 31 40 \
--max_epoch 50 > ./log/xunfei_train_0822_1.txt  

### 3.3训练模型3  
CUDA_VISIBLE_DEVICES=0 python train_LPRNet_xunfei6.py \
--pretrained_model './weights/Final_LPRNet_model.pth' \
--train_img_dirs '../xfdata/xunfei_train_ccpd2019_base_val_test_crpd.txt' \
--test_img_dirs '../xfdata/xunfei_carplate_train_val.txt' \
--save_folder '../user_data/tmp_data/xunfei_train_0818_4/' \
--learning_rate 0.0001 \
--lr_schedule 31 40 \
--max_epoch 50 > ./log/xunfei_train_0818_4.txt  

### 3.4训练模型4  
CUDA_VISIBLE_DEVICES=0 python train_LPRNet_xunfei6.py \
--pretrained_model './weights/Final_LPRNet_model.pth' \
--train_img_dirs '../xfdata/xunfei_train_ccpd2019_base_val_test.txt' \
--test_img_dirs '../xfdata/xunfei_carplate_train_val.txt' \
--save_folder '../user_data/tmp_data/xunfei_train_0816_1/' \
--learning_rate 0.0001 \
--lr_schedule 31 40 \
--max_epoch 50 > ./log/xunfei_train_0816_1.txt  

## 四、推理模型：对应在code目录下运行 bash test_2.sh 则不必运行本节以下命令  
### 4.1、利用重新训练的模型复现结果  
### 推理1  
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/tmp_data/xunfei_train_0818_3/LPRNet__iteration_108000.pth' \
--pretrained_model_stn '../user_data/tmp_data/xunfei_train_0818_3/STNet__iteration_108000.pth' \
--save './xunfeisave/submit_0818_3_1a.csv'  
### 推理2  
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0822_1/LPRNet__iteration_110000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0822_1/STNet__iteration_110000.pth' \
--save './xunfeisave/submit_0822_2a.csv'  
### 推理3  
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0818_4/LPRNet__iteration_112000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0818_4/STNet__iteration_112000.pth' \
--save './xunfeisave/submit_0818_4a.csv'  
### 推理4  
CUDA_VISIBLE_DEVICES=0 python run_LPRNet_xunfei6_1.py \
--test_img_dirs '../xfdata/xunfei_carplate_test.txt' \
--pretrained_model '../user_data/model_data/xunfei_train_0816_1/LPRNet__iteration_108000.pth' \
--pretrained_model_stn '../user_data/model_data/xunfei_train_0816_1/STNet__iteration_108000.pth' \
--save './xunfeisave/submit_0816_1a.csv'  
### 4.2、模型集成，生成 prediction_result目录下result.csv 文件  
python mix_data4.py  

## 所有结束  

