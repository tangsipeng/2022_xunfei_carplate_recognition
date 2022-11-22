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
    #srcpath = '/data/tangsipeng/mycode/carplate/ocr_datasets/xunfei_carplate_game/ccpd2019_test'
    #datapath = '/data/tangsipeng/mycode/carplate/ocr_datasets/xunfei_carplate_game/ccpd2019_test/base_test_quadrangle/*/*.jpg'
    srcpath = '../xfdata'
    datapath = '../user_data/other_data/ccpd2019_cut/*/*.jpg'    
    train = []
    val = []
    all = glob.glob(datapath)
    random.shuffle(all)
    write(all, os.path.join(srcpath, 'xunfei_ccpd2019_base_val_test.txt'))

if __name__ == '__main__':
    gen_txt()
