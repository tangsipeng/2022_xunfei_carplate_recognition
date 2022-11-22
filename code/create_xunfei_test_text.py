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
    #srcpath = '/data/tangsipeng/mycode/carplate/ocr_datasets/xunfei_carplate_game'
    #datapath = '/data/tangsipeng/mycode/carplate/ocr_datasets/xunfei_carplate_game/test/*.jpg'
    srcpath = '../xfdata/'
    datapath = '../xfdata/test/*.jpg'
    train = []
    val = []
    all = glob.glob(datapath)
    write(all, os.path.join(srcpath, 'xunfei_carplate_test.txt'))

if __name__ == '__main__':
    gen_txt()
