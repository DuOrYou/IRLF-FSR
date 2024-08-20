# from fileinput import filename
import os
path_temp = '/data/infrared/dy815/AISR_project/dataset/traindata_AISR'
path_train = '/data/infrared/dy815/AISR_project/dataset/traindata_AISR/train_data'
path_val = '/data/infrared/dy815/AISR_project/dataset/traindata_AISR/val_data'
path_test = '/data/infrared/dy815/AISR_project/dataset/traindata_AISR/test_data'
res_list = os.listdir(path_train)#lr ,sr2x,sr3x,sr4x
filename_list = []
for i in range(len(res_list)):
    
    filename_list = os.listdir(os.path.join(path_train,res_list[i]))
    filename_list.sort()
    for k in range(len(filename_list)):
        # print('copy data to train file')
        # aaa = str('%04d' % (k//8+1))
        if k+1<=1000*8 :
            pass
            # file_ = os.path.join(path_temp,res_list[i],filename_list[k])
            # newfile = os.path.join(path_train,res_list[i])
            # os.system('cp {0} {1}'.format(file_, newfile))
            
        elif k+1>len(filename_list)-34*8 :
            # print('copy data to test file')
            file_ = os.path.join(path_train,res_list[i],filename_list[k])
            newfile = os.path.join(path_test,res_list[i])
            if not os.path.exists(newfile):
                os.mkdir(newfile)
            os.system('cp {0} {1}'.format(file_, newfile))
            
        else:
            # print('copy data to val file')
            file_ = os.path.join(path_train,res_list[i],filename_list[k])
            newfile = os.path.join(path_val,res_list[i])
            if not os.path.exists(newfile):
                os.mkdir(newfile)
            os.system('cp {0} {1}'.format(file_, newfile))