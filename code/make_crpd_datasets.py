import os
import glob

def run():
    src = '../user_data/other_data/CRPD_cut'
    des = '../xfdata/CRPD_cut.txt'
    f=open(des,'w')
    for one in glob.glob(src+'/*/*.jpg'):
        filename = os.path.basename(one)
        if '警' in filename:
            continue
        if '字' in filename:
            continue
        if '挂' in filename:
            continue
        if '学' in filename:
            continue
        f.write(one+'\t'+filename+'\n')
    f.close()
    
if __name__ == '__main__':
    run()
