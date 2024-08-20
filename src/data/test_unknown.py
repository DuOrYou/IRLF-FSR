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
        self.video_dir = args.test_dir
        self.scale = args.scale
        self.patch_size = args.patch_size
        self.train = (not args.test_only)
        self.repeat = args.epochs
        self.angular = args.angular
        self.is_gray = args.is_gray_scale
        self.scale = args.scale
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        # self.test_hr_name = []
        self.test_lr_name= []
        self.name_list=[]
        # hr_dir = os.path.join(self.video_dir,'{}x'.format(self.scale),'HR')
        lr_dir = os.path.join(self.video_dir,'{}x'.format(self.scale),'HR')
        # hr_list = os.listdir(hr_dir)
        lr_list = os.listdir(lr_dir)
        for i in range(len(lr_list)):
            self.test_lr_name.append(os.path.join(lr_dir,lr_list[i] ))  
            # self.test_hr_name.append(os.path.join(hr_dir,hr_list[i] )) 
            
        # self.test_hr_name.sort()
        self.test_lr_name.sort()
    
    def __getitem__(self, idx):
        
        lr = self._load_file(idx)
        # filename = self.test_lr_name[idx]
        *_,filename = self.test_lr_name[idx].split('/')
        filename,_ = filename.split('.')
        
        # hr = hr.astype(np.float32)/65535.0
        # hr = torch.from_numpy(hr.astype(np.float32)/65535.0)
        
        lr = torch.from_numpy(lr.astype(np.float32)/255.0)
        # print('done load {}'.format(self.test_lr_name[idx]))
        return lr,filename

    def __len__(self):
        if self.train:
            return len(self.test_lr_name) 
        else:
            return len(self.test_lr_name)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.test_lr_name)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        # f_hr4x_ = self.train_hr4x_name[idx]#[h,w,ah,aw]
        # f_hr3x_ = self.train_hr3x_name[idx]
        # f_hr2x_ = self.train_hr2x_name[idx]
        f_lr_ = self.test_lr_name[idx]
        with open(f_lr_, 'rb') as _f:
            # print('loading hr image{}'.format(f_hr))
            f_lr = pickle.load(_f)
        # f_hr_ = self.test_hr_name[idx]
        # f_lr = cv2.imread(f_lr_,-1)
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

        return f_lr