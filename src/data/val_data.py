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
class val_process(data.Dataset):
    
    def __init__(self, args):
        self.val_dir = args.test_dir
        self.scale = args.scale
        self.patch_size = args.patch_size
        self.train = (not args.test_only)
        self.repeat = args.epochs
        self.angular = args.angular
        self.is_gray = args.is_gray_scale
        self.scale = args.scale
        self.val_hr_name = []
        self.val_lr_name= []
        self.name_list=[]
        hr_dir = os.path.join(self.val_dir,'{}x'.format(self.scale),'HR')
        lr_dir = os.path.join(self.val_dir,'{}x'.format(self.scale),'LR')
        hr_list = os.listdir(hr_dir)
        lr_list = os.listdir(lr_dir)
        for i in range(len(hr_list)):
            self.val_hr_name.append(os.path.join(hr_dir,hr_list[i] ))  
            self.val_lr_name.append(os.path.join(lr_dir,lr_list[i] )) 
            
        self.val_hr_name.sort()
        self.val_lr_name.sort()
        
    def __getitem__(self, idx):
        
        lr,hr = self._load_file(idx)
        *_,filename = self.val_lr_name[idx].split('/')
        filename,_ = filename.split('.')
        hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        lr = torch.from_numpy(lr.astype(np.float32)/255.0) 
        return lr,hr,filename

    def __len__(self):
        if self.train:
            return len(self.val_lr_name) 
        else:
            return len(self.val_lr_name)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.val_lr_name)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        # f_hr4x_ = self.train_hr4x_name[idx]#[h,w,ah,aw]
        # f_hr3x_ = self.train_hr3x_name[idx]
        # f_hr2x_ = self.train_hr2x_name[idx]
        f_lr_ = self.val_lr_name[idx]
        f_hr_ = self.val_hr_name[idx]
       
        with open(f_lr_, 'rb') as _f:
            # print('loading hr image{}'.format(f_hr))
            f_lr = pickle.load(_f)
        with open(f_hr_, 'rb') as _f1:
            # print('getting lr image{}'.format(f_lr))
            f_hr = pickle.load(_f1)
        # with open(f_hr2x_, 'rb') as _f2:
        #     # print('getting lr image{}'.format(f_lr))
        #     f_hr2x = pickle.load(_f2)
        # with open(f_lr_, 'rb') as _f3:
        #     # print('getting lr image{}'.format(f_lr))
        #     f_lr = pickle.load(_f3)

        return f_lr,f_hr