import os
import glob
import random

def write(data, des_path):
    f=open(des_path,'w')
    for one in data:
        label = os.path.basename(one)[:-4]
        f.write(one+'\t'+label+'\n')
    f.close()
    return 

def gen_txt():
    srcpath = '../xfdata/'
    datapath = '../xfdata/train/*.jpg'
    train = []
    val = []
    all = glob.glob(datapath)
    random.shuffle(all)
    all_len = len(all)
    train = all[:int(0.8*all_len)]
    val = all[int(0.8*all_len):]
    write(train, os.path.join(srcpath, 'xunfei_carplate_train.txt'))
    write(val, os.path.join(srcpath, 'xunfei_carplate_train_val.txt'))

if __name__ == '__main__':
    gen_txt()
