# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from data.load_data_xunfei_test2 import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
from model.STN import STNet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default="./data/test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=100, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model_stn', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--save', default="./xunfeisave/submit.csv", help='the test images path')
    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    imgnames = []
    for _, sample in enumerate(batch):
        imgname, img = sample
        imgs.append(img)
        imgnames.append(imgname)
    return (imgnames, torch.stack(imgs, 0))

def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    #
    STN = STNet()
    STN.to(device)    
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
        STN.load_state_dict(torch.load(args.pretrained_model_stn, map_location=lambda storage, loc: storage))
        print("STN loaded")          
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(lprnet, STN, test_dataset, args)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, STN, datasets, args):
    # TestNet = Net.eval()
    STN = STN.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    f = open(args.save,'w')
    save_result = []
    for i in range(epoch_size):
        # load train data
        filenames, images = next(batch_iterator)
        start = 0
        #targets = []
        #for length in lengths:
        #    label = labels[start:start+length]
        #    targets.append(label)
        #    start += length
        #targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        tansfer = STN(images)
        prebs = Net(tansfer)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        #print(prebs.shape) #100,68,18
        #print(prebs)
        preb_labels = list()
        #preb_labels_pp = list()
        
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            preb_label_pp = list()
            for j in range(preb.shape[1]): #68
                preb_label.append(np.argmax(preb[:, j], axis=0))
                #print('np.argmax',np.argmax(preb[:, j], axis=0))
                #print('np.max',np.max(preb[:, j], axis=0))
                softmax = torch.nn.Softmax(dim=0)(torch.tensor(preb[:, j]))
                preb_label_pp.append(np.max(softmax.numpy(), axis=0))
                #print(softmax)
                #print(np.max(softmax.numpy(), axis=0))
                #preb_pp.append(np.max(preb[:, j], axis=0))
                #print()
            #print(preb_pp)
            #print('len preb_label, preb_label_pp',len(preb_label),len(preb_label_pp))
            no_repeat_blank_label = list()
            no_repeat_blank_label_pp = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
                no_repeat_blank_label_pp.append(str(preb_label_pp[0]))
            for idx, c in enumerate(preb_label): # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                no_repeat_blank_label_pp.append(str(preb_label_pp[idx]))
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
            #preb_labels_pp.append(no_repeat_blank_label_pp)
            #
            lb = ""
            for idx in no_repeat_blank_label:
                lb += CHARS[idx]
            #print(filenames)
            #print(os.path.basename(filenames[i])+','+str(lb)+','+','.join(no_repeat_blank_label_pp))
            #f.write(os.path.basename(filenames[i])+','+str(lb)+','+','.join(no_repeat_blank_label_pp)+'\n')
            tmp = [os.path.basename(filenames[i]),str(lb)]
            tmp.append(','.join(no_repeat_blank_label_pp))
            save_result.append(tmp)
    save_result.sort(key=lambda i:i[0],reverse=True)
    for one in save_result:
        f.write(one[0]+','+one[1]+','+one[2]+'\n')
    f.close()    
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))



if __name__ == "__main__":
    test()

