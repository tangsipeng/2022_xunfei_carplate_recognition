import os
import numpy as np
#from numpy import *


def read(file):
    mydata_label = {}
    mydata_score = {}
    lines = open(file).readlines()
    for line in lines:
        line = line.strip()
        txts = line.split(',')
        imgname = txts[0]
        label = txts[1]
        score = txts[2:]
        mydata_label[imgname] = label
        new_score = []
        for one in score:
            new_score.append(float(one))
        mydata_score[imgname] = new_score
    return mydata_label, mydata_score

def run(file1,file2,file3,file4):
    final = []
    mydata_label1, mydata_score1 = read(file1)
    mydata_label2, mydata_score2 = read(file2)
    mydata_label3, mydata_score3 = read(file3)
    mydata_label4, mydata_score4 = read(file4)
    
    for key,value in mydata_label1.items():
        same_num = 0
        for sub_value in [mydata_label2[key],mydata_label3[key],mydata_label4[key]]:
            if value == sub_value:
                same_num+=1
        if  same_num >=2:
            final.append([key,value])
        else:
            #m1 = np.mean(mydata_score1[key])
            m2 = np.mean(mydata_score2[key])
            m3 = np.mean(mydata_score3[key])
            m4 = np.mean(mydata_score4[key])
            m_list = [m2,m3,m4]
            value_list = [mydata_label2[key],mydata_label3[key],mydata_label4[key]]
            idx = np.argmax(m_list)
            final.append([key,value_list[idx]])
    f=open('../prediction_result/result.csv','w')
    for one in final:
        imgname, label = one
        f.write(imgname+','+label+'\n')
    f.close()



if __name__ == '__main__':
    #file1 = 'xunfeisave/submit_0818_3_1a.csv'
    #file2 = 'xunfeisave/submit_0818_4a.csv'
    #0.9904
    #file1 = 'xunfeisave/submit_0818_3_1a.csv'
    #file2 = 'xunfeisave/submit_0822_1aa.csv'
    #file3 = 'xunfeisave/submit_0818_4a.csv'
    #0.9918
    file1 = 'xunfeisave/submit_0818_3_1a.csv'
    file2 = 'xunfeisave/submit_0822_2a.csv'
    file3 = 'xunfeisave/submit_0818_4a.csv'
    file4 = 'xunfeisave/submit_0816_1a.csv'
    run(file1,file2,file3,file4)

