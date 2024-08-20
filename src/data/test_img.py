from fileinput import filename
import os
from re import L
from turtle import shape
import torch.utils.data as data
import torch
import numpy as np
import random
import cv2
import pickle

from utility import *

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

class test_process(data.Dataset):
    
    def __init__(self, args):
        self.video_dir = args.video_dir
        self.scale = args.scale
        self.patch_size = args.patch_size
        self.train = (not args.test_only)
        self.repeat = args.epochs
        self.angular = args.angular
        self.is_gray = args.is_gray_scale
        self.scale = args.scale
        self.input_resolution = [288,384]
        self.test_hr_name = []
        self.file_name_list= []
        self.name_list=[]
        # hr_dir = os.path.join(self.video_dir,'{}x'.format(self.scale),'HR')
        # lr_dir = os.path.join(self.video_dir,'{}x'.format(self.scale),'LR')
        # hr_list = os.listdir(hr_dir)
        self.file_list = os.listdir(self.video_dir)
        # for i in range(len(self.file_list)):
        #     subdir = self.file_list[i]
        #     subdirlist = os.listdir(os.path.join(self.video_dir,subdir ))
        #     for j in range(len(subdirlist)):

        #         self.file_name_list.append(os.path.join(self.video_dir,self.file_list[i] , subdirlist[j]))  

        # for i in range(len(self.file_list)):
        #     subdir = self.file_list[i]
        #     subdirlist = os.listdir(os.path.join(self.video_dir,subdir ))
        #     for j in range(len(subdirlist)):

        #         self.file_name_list.append(os.path.join(self.video_dir,self.file_list[i],subdirlist[j] ))  

        for i in range(len(self.file_list)):
            subdir = self.file_list[i]
            # subdirlist = os.listdir(os.path.join(self.video_dir,subdir ))
            # for j in range(len(subdirlist)):

            self.file_name_list.append(os.path.join(self.video_dir,subdir ))  

            # self.test_hr_name.append(os.path.join(hr_dir,hr_list[i] )) 
        print(self.file_name_list)
        # self.test_hr_name.sort()
        self.file_name_list.sort()
        new_list = []
        if args.defaultname is not None:
            for s in self.file_name_list:
                if s.find(args.defaultname) >=0:
                    new_list.append(s) 
            self.file_name_list = new_list
        aa = self.file_name_list
        

    def __getitem__(self, idx):
        
        lr = self.load_imgs(idx)
        # filename = self.test_lr_name[idx]
        *_,filename = self.file_name_list[idx*self.angular**2].split('/')
        # print(self.file_name_list[idx*self.angular**2])
        filename,_ = filename.split('_')
        # hr = np.zeros_like(lr)
        # hr = hr.astype(np.float32)/65535.0
        # hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        print(lr.max())
        lr = torch.from_numpy(lr.astype(np.float32)/255.0) 
        hr = np.zeros_like(lr)
        hr = np.repeat(np.repeat(hr,self.scale,1),self.scale,0)
        return lr,hr,filename

    def __len__(self):
        if self.train:
            return len(self.file_name_list) //self.angular**2
        else:
            return len(self.file_name_list)  //self.angular**2
    
    def _get_index(self, idx):
        if self.train:
            return idx % (len(self.file_name_list) //self.angular**2)
        else:
            return idx% (len(self.file_name_list) //self.angular**2)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        # f_hr4x_ = self.train_hr4x_name[idx]#[h,w,ah,aw]
        # f_hr3x_ = self.train_hr3x_name[idx]
        # f_hr2x_ = self.train_hr2x_name[idx]
        f_lr_ = self.file_name_list[idx]
        # f_hr_ = self.test_hr_name[idx]
        f_lr = cv2.imread(f_lr_,-1)
        # f_hr = cv2.imread(f_hr_,-1)
        # if self.scale ==2:
        #     # print('getting lr2 image{}'.format(self.train_lr2_name[idx]))
        #     f_hr_ = self.train_hr2x_name[idx]#[h//scale,w//scale,ah,aw]
        # elif self.scale ==3:
        #     f_hr_ = self.train_hr3x_name[idx]
        # elif self.scale ==4:
        #     f_hr_ = self.train_hr4x_name[idx]
       
        # with open(f_lr_, 'rb') as _f:
        #     # print('loading hr image{}'.format(f_hr))
        #     f_lr = pickle.load(_f)
        # with open(f_hr_, 'rb') as _f1:
        #     # print('getting lr image{}'.format(f_lr))
        #     f_hr = pickle.load(_f1)
        # with open(f_hr2x_, 'rb') as _f2:
        #     # print('getting lr image{}'.format(f_lr))
        #     f_hr2x = pickle.load(_f2)
        # with open(f_lr_, 'rb') as _f3:
        #     # print('getting lr image{}'.format(f_lr))
        #     f_lr = pickle.load(_f3)

        return f_lr,f_lr_
    
    def load_imgs(self,idx):
            
            aa = self.angular**2
            idx = self._get_index(idx)
            count =0
            # print('Making a binary for img : {}'.format(count))
            img_original = np.zeros([self.input_resolution[0],self.input_resolution[1],self.angular,self.angular])
            if self.is_gray:
                for u in range(self.angular):
                    for v in range(self.angular):
                        img_original[:,:,u,v] = cv2.imread(self.file_name_list[idx*self.angular**2+count],cv2.IMREAD_GRAYSCALE)
                        count +=1
            else:
                for u in range(self.angular):
                    for v in range(self.angular):
                        img_original[:,:,u,v] = cv2.imread(self.file_name_list[idx*self.angular**2+count],cv2.IMREAD_GRAYSCALE)
                        count +=1
            # print(self.file_name_list[idx*aa+ 6])
            # img_up = np.stack([img1,img2,img3],axis = -1)
            # img_center = np.stack([img4,img5,img6],axis = -1)
            # img_down = np.stack([img7,img8,img9],axis = -1)
            # img_original = np.stack([img_up,img_center,img_down],axis = -1) # [h,w,ah,aw] for gray image , [h,w,3,ah,aw]
            # # print(idx)

            return img_original

            