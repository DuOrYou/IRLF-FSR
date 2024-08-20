from logging import raiseExceptions
from math import ceil
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

def add_stripe(im, beta = None):

    # stdN_G = np.random.uniform(0.03, 0.05)    #控制高斯噪声强度
    stdN_G = 0.01
    noise_G= np.random.normal(0, stdN_G* 255, im.shape)

    if beta is not None:
        beta = beta
    else:
        beta = np.random.uniform(0.03, 0.05)  #控制条纹噪声强度

    noise_col = np.random.normal(0, beta * 255. , im.shape[1])
    S_noise = np.tile(noise_col, (im.shape[0], 1))
    # S_noise = np.rot90(S_noise,1)
    # if np.random.rand(1)>0.5: ## 添加竖条纹
    #     noise_col = np.random.normal(0, beta, im.shape[1])
    #     S_noise = np.tile(noise_col, (im.shape[0], 1))
    # else:
    #     noise_row = np.random.normal(0, beta, im.shape[0])
    #     S_noise = np.tile(noise_row, (1, im.shape[1]))
    # print(S_noise)

    out = im+S_noise +noise_G

    return np.clip(out,0.,255.)



class train_process(data.Dataset):
    
    def __init__(self, args):
        self.train_dir = args.train_dir
        self.raw_dir = args.original_data
        self.scale = args.scale
        # self.save_path = args.train_savepath
        self.gen_train_data = args.gen_train_data
        self.patch_size = args.patch_size
        self.augment = args.data_augment
        self.train = (not args.test_only)
        self.repeat = args.epochs
        self.angular = args.angular
        self.crop_times = args.crop_times
        self.is_gray = args.is_gray_scale
        self.val_dir = args.val_dir
        self.test_dir = args.test_dir
        self.add_nonuniform_N = args.nonuniform_N
        self.train_hr_name = []
        self.train_lr_name = []
        noise_list = args.noise_level
        if self.gen_train_data:
            if not os.path.exists(self.train_dir):
                os.makedirs(self.train_dir)
            print('start to generating train data ....')
            # count  = 0
            
            folder_list = os.listdir(self.raw_dir)
            
            for noise_num in range(len(noise_list)):
                noise_lv_ = noise_list[noise_num]
                
                file_name_list = []
                for i in range(len(folder_list)):
                    # print('this file is {} '.format(folder_list[i]))
                    # file_name_list_ = []
                    name_list = os.listdir(os.path.join(self.raw_dir,folder_list[i]))#image names
                    for names in range(len(name_list)):
                        
                        file_name_list.append(os.path.join(self.raw_dir,folder_list[i],name_list[names]) ) 
                    # file_name_list_.sort()
                file_name_list.sort()
                print('this file has {} pairs of image'.format(len(file_name_list)//self.angular**2))
                print('{} for train,{} for val,{} for test'.format(ceil((len(file_name_list)//self.angular**2)*0.5),
                                                                ceil((len(file_name_list)//self.angular**2)*0.8)-ceil((len(file_name_list)//self.angular**2)*0.5),
                                                                len(file_name_list)//self.angular**2-ceil((len(file_name_list)//self.angular**2)*0.8)))
                for k in range(len(file_name_list)//self.angular**2):
                    # count += 1
                    # print('Making a binary for img : {}'.format(count))
                    img_original = np.zeros([480,640,self.angular,self.angular])
                    count = 0
                    if self.is_gray:
                        for u in range(self.angular):
                            for v in range(self.angular):
                                img_original[:,:,u,v] = cv2.imread(file_name_list[k*self.angular**2+count],cv2.IMREAD_GRAYSCALE)
                                count +=1
                        
                    else:
                        for u in range(self.angular):
                            for v in range(self.angular):
                                img_original[:,:,u,v] = cv2.imread(file_name_list[k*self.angular**2+count],cv2.IMREAD_GRAYSCALE)
                                count +=1
                    *_,kinds,name = file_name_list[k*self.angular**2+0].split('/')
                    # img_num,_ = name.split('_')
                    patchidx = kinds
                    # patchidx = kinds +'_'+img_num
                    # print(patchidx)
                    # img_up = np.stack([img1,img2,img3],axis = -1)
                    # img_center = np.stack([img4,img5,img6],axis = -1)
                    # img_down = np.stack([img7,img8,img9],axis = -1)
                    # img_original = np.stack([img_up,img_center,img_down],axis = -1) # [h,w,ah,aw] for gray image , [h,w,3,ah,aw]
                    # print(self.patch_size*4)
                    # if ceil((len(file_name_list)//self.angular**2)*0.5)<k:
                    #     x = 0
                    #     crop_num = '%02d' % x
                    #         # size =self.patch_size
                    #     self.train_hr = img_original
                    #         # print('hr shape is : {}'.format(self.train_hr.shape))
                    #     h = self.train_hr.shape[0]
                    #     w = self.train_hr.shape[1]
                        
                    #     # self.hr2x = self.train_hr[:h,:w,:,:]
                    #     # self.hr3x = self.train_hr[:h,:w,:,:]
                    #     # self.hr4x = self.train_hr[:h,:w,:,:]
                    #     self.hr8x = self.train_hr[:h,:w,:,:]
                        
                    #     # self.lr4x = np.zeros_like(self.hr2x)[:h//4,:w//4,:,:]
                    #     # self.lr3x = np.zeros_like(self.hr2x)[:h//3,:w//3,:,:]
                    #     self.lr8x = np.zeros_like(self.hr8x)[:h//self.scale,:w//self.scale,:,:]
                    #     # print('lr_4 shape is : {}'.format(self.lr_4.shape))
                    #     for o in range (self.hr8x.shape[2]):
                    #         for p in range(self.hr8x.shape[3]):

                    #             temp_8x = cv2.resize(self.hr8x[:,:,o,p],dsize=None,fx=1/self.scale,fy=1/self.scale, interpolation=cv2.INTER_CUBIC)
                    #             # temp_3x = cv2.resize(self.hr3x[:,:,o,p],dsize=None,fx=1/3,fy=1/3, interpolation=cv2.INTER_CUBIC)
                    #             # temp_4x = cv2.resize(self.hr4x[:,:,o,p],dsize=None,fx=1/4,fy=1/4, interpolation=cv2.INTER_CUBIC)
                    #             if self.add_nonuniform_N:
                    #                 # temp_2x = add_stripe(temp_2x,noise_lv_)
                    #                 # temp_3x = add_stripe(temp_3x,noise_lv_)
                    #                 temp_8x = add_stripe(temp_8x,noise_lv_)

                    #             self.lr8x[:,:,o,p] = temp_8x
                    #             # self.lr3x[:,:,o,p] = temp_3x
                    #             # self.lr4x[:,:,o,p] = temp_4x
                    #             # self.lr2x[:,:,o,p] = cv2.resize(self.hr2x[:,:,o,p],dsize=None,fx=1/2,fy=1/2, interpolation=cv2.INTER_CUBIC)
                    #             # self.lr3x[:,:,o,p] = cv2.resize(self.hr3x[:,:,o,p],dsize=None,fx=1/3,fy=1/3, interpolation=cv2.INTER_CUBIC)
                    #             # self.lr4x[:,:,o,p] = cv2.resize(self.hr4x[:,:,o,p],dsize=None,fx=1/4,fy=1/4, interpolation=cv2.INTER_CUBIC)
                    #     # patchidx = '%04d' % count 
                    #     print(self.lr8x.shape)
                    #     self._save_data(patchidx,crop_num,save_stat = 'test',noise_lv = noise_lv_)
                    # else:
                    for x in range(self.crop_times):
                        crop_num = '%02d' % x
                        # size =self.patch_size
                        
                        self.hr8x = self.crop_patch(self.patch_size*self.scale, img_original)
                        self.lr8x = np.zeros_like(self.hr8x)[:self.patch_size,:self.patch_size,:,:]
                        # print('hr shape is : {}'.format(self.train_hr.shape))
                    
                        for o in range (self.hr8x.shape[2]):
                            for p in range(self.hr8x.shape[3]):
                                temp_8x = cv2.resize(self.hr8x[:,:,o,p],dsize=None,fx=1/self.scale,fy=1/self.scale, interpolation=cv2.INTER_CUBIC)
                                
                                if self.add_nonuniform_N:
                                    temp_8x = add_stripe(temp_8x)
                                    

                                self.lr8x[:,:,o,p] = temp_8x
                        
                        # patchidx = '%04d' % count 
                        self._save_data(patchidx,crop_num,save_stat = 'train')
        else:
            
            hr_dir = os.path.join(self.train_dir,'{}x'.format(self.scale),'HR')
            lr_dir = os.path.join(self.train_dir,'{}x'.format(self.scale),'LR')
            hr_list = os.listdir(hr_dir)
            lr_list = os.listdir(lr_dir)
            for i in range(len(hr_list)):
                self.train_hr_name.append(os.path.join(hr_dir,hr_list[i] ))  
                self.train_lr_name.append(os.path.join(lr_dir,lr_list[i] ))
            self.train_lr_name.sort()
            self.train_hr_name.sort()
        
    def crop_patch(self,patch_size,imgs):
        # print(imgs.shape[0]-patch_size)
        # print(imgs.shape[1]-patch_size)
        x = random.randrange(0,imgs.shape[0]-patch_size,8)
        y = random.randrange(0,imgs.shape[1]-patch_size,16)
        img_patch = imgs[x:x+patch_size,y:y+patch_size,:,:]#[patch_size,patch_size,ah,aw]
        return img_patch

    def _save_data(self,patchidx,num,save_stat,noise_lv = None):
        if save_stat == 'train':
            if not os.path.exists(self.train_dir):
                os.makedirs(self.train_dir)
                
            # filedir_hr4x = os.path.join(self.train_dir,'4x','HR' )
            # filedir_lr4x = os.path.join(self.train_dir,'4x','LR' )
            # if not os.path.exists(filedir_hr4x):
            #     os.makedirs(filedir_hr4x)
            #     os.makedirs(filedir_lr4x)
                
            # filedir_hr3x = os.path.join(self.train_dir,'3x','HR' )
            # filedir_lr3x = os.path.join(self.train_dir,'3x','LR' )
            # if not os.path.exists(filedir_hr3x):
            #     os.makedirs(filedir_hr3x)
            #     os.makedirs(filedir_lr3x)
            
            
            # filedir_hr2x = os.path.join(self.train_dir,'2x','HR' )
            # filedir_lr2x = os.path.join(self.train_dir,'2x','LR' )
            # if not os.path.exists(filedir_hr2x):
            #     os.makedirs(filedir_hr2x)
            #     os.makedirs(filedir_lr2x)

            filedir_hr8x = os.path.join(self.train_dir,'{}x'.format(self.scale),'HR' )
            filedir_lr8x = os.path.join(self.train_dir,'{}x'.format(self.scale),'LR' )
            if not os.path.exists(filedir_hr8x):
                os.makedirs(filedir_hr8x)
                os.makedirs(filedir_lr8x)
            
        elif save_stat == 'val':
            if not os.path.exists(self.val_dir):
                os.makedirs(self.val_dir)
            
            filedir_hr8x = os.path.join(self.val_dir,'{}x'.format(self.scale),'HR' )
            filedir_lr8x = os.path.join(self.val_dir,'{}x'.format(self.scale),'LR' )
            if not os.path.exists(filedir_hr8x):
                os.makedirs(filedir_hr8x)
                os.makedirs(filedir_lr8x)

            # filedir_hr4x = os.path.join(self.val_dir,'4x','HR' )
            # filedir_lr4x = os.path.join(self.val_dir,'4x','LR' )
            # if not os.path.exists(filedir_hr4x):
            #     os.makedirs(filedir_hr4x)
            #     os.makedirs(filedir_lr4x)
                
            # filedir_hr3x = os.path.join(self.val_dir,'3x','HR' )
            # filedir_lr3x = os.path.join(self.val_dir,'3x','LR' )
            # if not os.path.exists(filedir_hr3x):
            #     os.makedirs(filedir_hr3x)
            #     os.makedirs(filedir_lr3x)
    
            # filedir_hr2x = os.path.join(self.val_dir,'2x','HR' )
            # filedir_lr2x = os.path.join(self.val_dir,'2x','LR' )
            # if not os.path.exists(filedir_hr2x):
            #     os.makedirs(filedir_hr2x)
            #     os.makedirs(filedir_lr2x)
                
        elif save_stat == 'test':
            if not os.path.exists(self.test_dir):
                os.makedirs(self.test_dir)
                
            filedir_hr8x = os.path.join(self.test_dir,'{}xS-{}'.format(self.scale,noise_lv),'HR' )
            filedir_lr8x = os.path.join(self.test_dir,'{}xS-{}'.format(self.scale,noise_lv),'LR' )
            if not os.path.exists(filedir_hr8x):
                os.makedirs(filedir_hr8x)
                os.makedirs(filedir_lr8x)
                
            # filedir_hr3x = os.path.join(self.test_dir,'3xS-{}'.format(noise_lv),'HR' )
            # filedir_lr3x = os.path.join(self.test_dir,'3xS-{}'.format(noise_lv),'LR' )
            # if not os.path.exists(filedir_hr3x):
            #     os.makedirs(filedir_hr3x)
            #     os.makedirs(filedir_lr3x)
    
            # filedir_hr2x = os.path.join(self.test_dir,'2xS-{}'.format(noise_lv),'HR' )
            # filedir_lr2x = os.path.join(self.test_dir,'2xS-{}'.format(noise_lv),'LR' )
            # if not os.path.exists(filedir_hr2x):
            #     os.makedirs(filedir_hr2x)
            #     os.makedirs(filedir_lr2x)
        
        filename_hr8x = os.path.join(filedir_hr8x,str(patchidx)+'_c'+str(num)+'.pt')
        filename_lr8x = os.path.join(filedir_lr8x,str(patchidx)+'_c'+str(num)+'.pt')

        # filename_hr4x = os.path.join(filedir_hr4x,str(patchidx)+'_c'+str(num)+'.pt')
        # filename_lr4x = os.path.join(filedir_lr4x,str(patchidx)+'_c'+str(num)+'.pt')
        
        # filename_hr3x = os.path.join(filedir_hr3x,str(patchidx)+'_c'+str(num)+'.pt')
        # filename_lr3x = os.path.join(filedir_lr3x,str(patchidx)+'_c'+str(num)+'.pt')
        
        # filename_hr2x = os.path.join(filedir_hr2x,str(patchidx)+'_c'+str(num)+'.pt')
        # filename_lr2x = os.path.join(filedir_lr2x,str(patchidx)+'_c'+str(num)+'.pt')
        if self.scale == 1:
            self.train_hr_name.append(filename_hr8x)
            self.train_lr_name.append(filename_lr8x)
        # elif self.scale ==4:
        #     self.train_hr_name.append(filename_hr4x)
        #     self.train_lr_name.append(filename_lr4x)
        # elif self.scale ==3:
        #     self.train_hr_name.append(filename_hr3x)
        #     self.train_lr_name.append(filename_lr3x)
        else:
            raise ValueError('scale must be one of [2,3,4] but got {}'.format(self.scale)) 
        
        if not os.path.isfile(filename_hr8x):
            with open(filename_hr8x,'wb') as _f:
                pickle.dump(self.hr8x, _f)

        if not os.path.isfile(filename_lr8x):
            print(filename_lr8x)
            print(self.lr8x.shape)
            with open(filename_lr8x,'wb') as _f:
                pickle.dump(self.lr8x, _f)



        # if not os.path.isfile(filename_hr4x):
        #     with open(filename_hr4x,'wb') as _f:
        #         pickle.dump(self.hr4x, _f)
                
        # if not os.path.isfile(filename_hr3x):    
        #     with open(filename_hr3x,'wb') as _f1:
        #         pickle.dump(self.hr3x,_f1)
                
        # if not os.path.isfile(filename_hr2x):
        #     with open(filename_hr2x,'wb') as _f2:
        #         pickle.dump(self.hr2x,_f2)
                
        # if not os.path.isfile(filename_lr4x):
        #     with open(filename_lr4x,'wb') as _f:
        #         pickle.dump(self.lr4x, _f)
                
        # if not os.path.isfile(filename_lr3x):    
        #     with open(filename_lr3x,'wb') as _f1:
        #         pickle.dump(self.lr3x,_f1)
                
        # if not os.path.isfile(filename_lr2x):
        #     with open(filename_lr2x,'wb') as _f2:
        #         pickle.dump(self.lr2x,_f2)

    def __getitem__(self, idx):
        
        lr,hr = self._load_file(idx)
        if self.augment:
            
            lr,hr = self.data_augment(lr,hr)
        hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        # hr = hr.contiguous().view(-1,hr.shape[0],hr.shape[1])
        
        # print(lr.min())
        lr = torch.from_numpy(lr.astype(np.float32)/255.0)  
        
        
        return lr,hr

    def __len__(self):
        if self.train:
            return len(self.train_hr_name)
        else:
            return len(self.train_hr_name)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.train_hr_name)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        # f_hr4x_ = self.train_hr4x_name[idx]#[h,w,ah,aw]
        # f_hr3x_ = self.train_hr3x_name[idx]
        # f_hr2x_ = self.train_hr2x_name[idx]
        
        f_lr_ = self.train_lr_name[idx]
        # print(f_lr_)
        f_hr_ = self.train_hr_name[idx]
    
        with open(f_lr_, 'rb') as _f:
            # print('loading hr image{}'.format(f_hr))
            f_lr = pickle.load(_f)
        with open(f_hr_, 'rb') as _f1:
            # print('getting lr image{}'.format(f_lr))
            f_hr = pickle.load(_f1)
        # print(f_hr.shape)
        # print('lr shape is {}'.format(f_lr.shape))
        return f_lr,f_hr
    
    def data_augment(self,lr,hr):
        # 4D augmentation
        # flip
        if np.random.rand(1)>0.5:
            lr = np.flip(np.flip(lr,0),2)
            hr = np.flip(np.flip(hr,0),2)
            # hr3x = np.flip(np.flip(hr3x,0),2)
            # hr4x = np.flip(np.flip(hr4x,0),2)
            # lr_8 = np.flip(np.flip(lr_8,0),2)                
        if np.random.rand(1)>0.5:
            lr = np.flip(np.flip(lr,1),3)
            hr = np.flip(np.flip(hr,1),3)
            # hr3x = np.flip(np.flip(hr3x,1),3)
            # hr4x = np.flip(np.flip(hr4x,1),3)
            # lr_8 = np.flip(np.flip(lr_8,1),3)
        # rotate
        r_ang = np.random.randint(1,5)
        hr = np.rot90(hr,r_ang,(2,3))
        hr = np.rot90(hr,r_ang,(0,1))
        
        # hr3x = np.rot90(hr3x,r_ang,(2,3))
        # hr3x = np.rot90(hr3x,r_ang,(0,1))
        
        # hr4x = np.rot90(hr4x,r_ang,(2,3))
        # hr4x = np.rot90(hr4x,r_ang,(0,1))
        
        lr = np.rot90(lr,r_ang,(2,3))
        lr = np.rot90(lr,r_ang,(0,1))           
        

        # to tensor     
         # [an,h,w]
         #[an,hs,ws]

        # hr2x = torch.from_numpy(hr2x.astype(np.float32)/255.0)
        # hr3x = torch.from_numpy(hr3x.astype(np.float32)/255.0)
        # hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        # hr = hr.contiguous().view(-1,hr.shape[0],hr.shape[1])
        
        
        # lr = torch.from_numpy(lr.astype(np.float32)/255.0)  
        # lr = lr.contiguous().view(-1,hr.shape[0],hr.shape[1])
        
        return lr,hr
def get_train_data():



    pass
    
if __name__ == '__main__':

    
    get_train_data()