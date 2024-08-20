# -*- coding: utf-8 -*-
import fractions
from genericpath import exists
import math
from math import ceil
from re import I
from tkinter import Y
from tkinter.ttk import Progressbar
from unittest import mock
import numpy as np
import cv2
import torch
import os
import cmath
import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
# from generate_PSF import PSF as PSF
import pickle
from scipy import signal
from torchvision import utils as vutils
import matplotlib as plt
import matplotlib.image as img

# from src.utility import LFsplit



def gen_noise_map(img,noise_type,noise_lv):
    img_ = img
    if noise_lv == 0:
        noise_map = np.zeros_like(img_,dtype=np.float32)
    else:        
        if noise_type == 'Gauss':
            noise_map = np.random.normal(0, noise_lv/255.0, img_.shape).astype(np.float32)
        elif noise_type == 'Poisson':
            noise_map = np.random.poisson(img_*noise_lv).astype(np.float32)/noise_lv -img_
    return noise_map

def add_noise(img_s,noise_map,range = [0.0,1.0]):
    val_min,val_max = range
    img_ = img_s
    noised = img_+noise_map
    
    out_img = np.clip(noised, val_min, val_max)
        
    return out_img

def down_sampling(img,dp_type=2, sf=4):
    '''dytype: interpolation typer of downsampling, [1=INTER_LINEAR],[2=cv2.INTER_CUBIC],[0=cv2.INTER_NEAREST]'''
    # if np.min(img)<0 or np.max(img)>1:
    #     raise ValueError('the value of image is in range {}-{}, which is not expected'.format(np.min(img),np.max(img)))
    img_ = img
    img_ = cv2.resize(img_,dsize=None,fx=1/sf,fy=1/sf, interpolation=dp_type)
    img_ = np.clip(img_, 0.0, 1.0)
    # if np.min(img)<0 or np.max(img)>1:
    #     raise ValueError('the value of image is in range {}-{}, which is not expected'.format(np.min(img),np.max(img)))
    return img_


def apply_degradation(img,kernel,noise_map,dp_scale,degradation = 2, shuffle = False):
    
    '''
    L = (H*K)⬇ + n 
    
    img: input image [h,w]
    kernel: blur kernel
    noise_map: added noise 
    dp_scale: downsampling scale of degradation,default 4
    shuffle: wether use random order while degradating image,default 'False'
    '''
    buffer = img
    if shuffle:
        orderdict = random.sample(range(3),3)
        
        for i in range(len(orderdict)):
            if orderdict[i]==0:
                buffer = ndimage.filters.convolve(buffer,kernel, mode='reflect')
            elif orderdict[i]==1:
                buffer = down_sampling(buffer,dp_type=degradation, sf = dp_scale)
                noise_map = down_sampling(noise_map,dp_type=degradation, sf = dp_scale) 
            else:
                buffer = add_noise(buffer,noise_map)
        
    else:
        
        buffer = ndimage.filters.convolve(buffer,kernel, mode='reflect')
        buffer = down_sampling(buffer,dp_type=degradation, sf = dp_scale)
        noise_map = down_sampling(noise_map,dp_type=degradation, sf = dp_scale) 
        buffer = add_noise(buffer,noise_map)
    out_img = buffer
    return out_img

def local_LFsplit(data, angRes):# split [B*C*AH*AW] to [B*(C*A*A)*H*W]
    H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = np.stack(data_sv, dim=-1)
    
    return data_st

def gen_my_dataset(img_stack,img_name_,save_path,Mode,b_gauss ,b_motion,b_unseen,crop_times = 16,sf=4,cropsize = 32,blur_type_=['anti_ISO','Motion'],
                   noise_type_ = ['Gauss','Poisson'],gauss_lv_ = [0,5,15,30,60],poisson_size_= [0,100,1000,10000],
                   degradation_= [2],shuffle = False):
    
    if shuffle:
        shuffle_name = 'shuffle'
    else:
        shuffle_name = 'unshuffle'
    aa = np.min(img_stack)
    if Mode =='train':
        for bl_num in range(len(blur_type_)):
            blur_type = blur_type_[bl_num]
            for ne_num in range(len(noise_type_)):
                noise_type = noise_type_[ne_num]
                
                for iter in range (crop_times):
                    
                    # noise_type = random.choice(noise_type_)
                    ### generate blur kernel 
                    if blur_type == 'simga':
                        blur_kernel = b_gauss[...,iter]
                    elif blur_type =='omsim':
                        blur_kernel = b_motion[...,iter]
                    elif blur_type == 'unsim':
                        blur_kernel = b_unseen[...,iter//2]
                    ### generate noise_map 
                    degradation = random.choice(degradation_)
                    
                    # noise_type = random.choice(noise_type_)
                    if noise_type == 'Gauss':
                        noise_lv = random.choice(gauss_lv_)
                    elif noise_type == 'Poisson':
                        noise_lv = random.choice(poisson_size_)
                    else:
                        raise ValueError('noise type [{}] is not defined'.format(noise_type))
                    
                    random_x = random.randrange(0, img_stack[:,:,0].shape[0]-cropsize*sf, cropsize)
                    random_y = random.randrange(0, img_stack[:,:,0].shape[1]-cropsize*sf, cropsize)
                    
                    degraded_imgs = []
                    imgs_gt = []
                    # dd = np.min(img_stack)
                    # if not aa ==dd:
                    #     raise ValueError('original data was changed')
                    for angRes in range(9):
                        img_s = img_stack[:,:,angRes]
                        assert np.min(img_stack) == aa, 'the input distribution is changed'
                        # ee = np.min(img_stack)
                        # if not aa ==ee:
                        #     raise ValueError('original data was changed')
                        img_gt = img_s[random_x:random_x+cropsize*sf,random_y:random_y+cropsize*sf]
                        imgs_gt.append(img_gt)
                        
                        noise_lv_s = noise_lv+(random.random()-0.5)*0.002*noise_lv
                        noise_map = gen_noise_map(img_gt,noise_type=noise_type,noise_lv=noise_lv_s)
                        degraded_img = apply_degradation(img_gt,kernel=blur_kernel,noise_map=noise_map,
                                                        dp_scale=sf,degradation=degradation,shuffle=shuffle)
                        
                        degraded_imgs.append(degraded_img)
                                
                    train_dir = os.path.join(save_path,Mode,'{}-x{}'.format(shuffle_name,sf),'{}-{}'.format(blur_type,noise_type))
                    
                    if not os.path.exists(train_dir):
                        os.makedirs(train_dir)
                    iter_ = '%02d' % iter
                    image_train_name = os.path.join(train_dir,img_name_+'_{}.pt'.format(iter_))
                    degraded_img_save = np.array(degraded_imgs)
                    imgs_gt_save = np.array(imgs_gt)
                    
                    imgs_gt_save = np.reshape(np.transpose(imgs_gt_save,(1,2,0)),[imgs_gt_save.shape[1],imgs_gt_save.shape[2],3,3])
                    degraded_img_save = np.reshape(np.transpose(degraded_img_save,(1,2,0)),[degraded_img_save.shape[1],degraded_img_save.shape[2],3,3])
                    
                    dataDict_train = {'img_lr_train':degraded_img_save,'img_hr_train':imgs_gt_save,'kernel_train':blur_kernel,'noise_lv':noise_lv}
                    
                    with open(image_train_name, "wb") as tf:
                        pickle.dump(dataDict_train,tf)
                
    elif Mode =='test':
        
        degradation = random.choice(degradation_)
        for j in range(len(blur_type_)):
            blur_type = blur_type_[j]
            if blur_type == 'simga':
                multi_kernel = b_gauss
            elif blur_type =='omsim':
                multi_kernel = b_motion
            elif blur_type == 'unsim':
                multi_kernel = b_unseen
            for k in range (multi_kernel.shape[-1]):
                current_kernel = multi_kernel[:,:,k] ## the blur kernel is defined
                
                for l in range(len(noise_type_)):
                    noise_type = noise_type_[l]## get noise type
                    if noise_type == 'Gauss':
                        noise_lv_ = gauss_lv_
                    elif noise_type == 'Poisson':
                        noise_lv_ = poisson_size_
                    else:
                        raise ValueError('noise type [{}] is not defined'.format(noise_type))
                    for m in range(len(noise_lv_)):
                        noise_lv = noise_lv_[m]
                        degraded_imgs = []
                        for angRes in range(9):
                            img_gt = img_stack[:,:,angRes]
                            assert np.min(img_stack) == aa, 'the input distribution is changed'
                            img_ = img_gt
                            noise_lv_s = noise_lv+(random.random()-0.5)*0.002*noise_lv
                            noise_map = gen_noise_map(img_,noise_type=noise_type,noise_lv=noise_lv_s)
                            degraded_img = apply_degradation(img_,kernel=current_kernel,noise_map=noise_map,
                                                    dp_scale=sf,degradation=degradation,shuffle=shuffle)
                            degraded_imgs.append(degraded_img)
                            
                        degraded_img_save = np.array(degraded_imgs)
                        degraded_img_save = np.reshape(np.transpose(degraded_img_save,(1,2,0)),[degraded_img_save.shape[1],degraded_img_save.shape[2],3,3])
                        
                        test_dir = os.path.join(save_path,Mode,'{}-x{}'.format(shuffle_name,sf),'{}-{}-{}-{}'.format(blur_type,k,noise_type,noise_lv))

                        if not os.path.exists(test_dir):
                            os.makedirs(test_dir)
                        dataDict_test = {'img_lr_test':degraded_img_save,'kernel_test':current_kernel,'noise_lv':noise_lv}
                        save_test_name = os.path.join(test_dir,img_name_+'.pt')
                        with open(save_test_name, "wb") as tf:
                            pickle.dump(dataDict_test,tf)
               
        # imgs_gt = []
        # for angRes in range(9):
        #     img_gt = img_stack[:,:,angRes]
        #     img_ = img_gt            
        #     imgs_gt.append(img_gt)
        
        # test_saved_imgs = np.array(imgs_gt)
        # test_saved_imgs = np.reshape(np.transpose(test_saved_imgs,(1,2,0)),[test_saved_imgs.shape[1],test_saved_imgs.shape[2],3,3])
        # test_dir_gt = os.path.join(save_path,Mode,'HR')
        # if not os.path.exists(test_dir_gt):
        #     os.makedirs(test_dir_gt)
        # save_gt_name = os.path.join(test_dir_gt,img_name_+'.pt')
        # with open(save_gt_name, "wb") as tf:
        #     pickle.dump(test_saved_imgs,tf)   
                                                         
    # else:
    #     assert np.min(img_stack) == aa
    #     kernelSize = kernelSize_[0]
    #     degradation = random.choice(degradation_)
    #     random_x = random.randrange(0, img_stack[:,:,0].shape[0]-cropsize*sf, cropsize)
    #     random_y = random.randrange(0, img_stack[:,:,0].shape[1]-cropsize*sf, cropsize)
    #     assert np.min(img_stack) == aa
    #     for j in range(len(blur_type_)):
    #         blur_type = blur_type_[j]
    #         assert np.min(img_stack) == aa
    #         multi_kernel = gen_multi_blur_kernel( kernelSize = kernelSize,blur_type = blur_type,multi_fractions =Motion_fraction_,anti_iso_list = kernel_width_)
    #         assert np.min(img_stack) == aa
    #         for k in range (multi_kernel.shape[0]):
    #             current_kernel = multi_kernel[k,:,:] ## the blur kernel is defined
                
    #             for l in range(len(noise_type_)):
    #                 noise_type = noise_type_[l]## get noise type
    #                 if noise_type == 'Gauss':
    #                     noise_lv_ = gauss_lv_
    #                 elif noise_type == 'Poisson':
    #                     noise_lv_ = poisson_size_
    #                 else:
    #                     raise ValueError('noise type [{}] is not defined'.format(noise_type))
    #                 assert np.min(img_stack) == aa
    #                 for m in range(len(noise_lv_)):
    #                     noise_lv = noise_lv_[m]
    #                     degraded_imgs = []
    #                     ground_truths = []
    #                     for angRes in range(9):
    #                         img_s = img_stack[:,:,angRes]
    #                         assert np.min(img_stack) == aa, 'the input distribution is changed'
    #                         img_ = img_s
    #                         img_gt = img_s[random_x:random_x+cropsize*sf,random_y:random_y+cropsize*sf]
    #                         noise_lv_s = noise_lv+(random.random()-0.5)*0.002*noise_lv
    #                         noise_map = gen_noise_map(img_gt,noise_type=noise_type,noise_lv=noise_lv_s)
    #                         assert np.min(img_stack) == aa
    #                         degraded_img = apply_degradation(img_gt,kernel=current_kernel,noise_map=noise_map,
    #                                                 dp_scale=sf,degradation=degradation,shuffle=shuffle)
    #                         assert np.min(img_stack) == aa
    #                         degraded_imgs.append(degraded_img)
    #                         ground_truths.append(img_gt)
                            
    #                     gt_img_save = np.array(ground_truths)
    #                     degraded_img_save = np.array(degraded_imgs)
    #                     val_img_save = np.reshape(np.transpose(gt_img_save,(1,2,0)),[gt_img_save.shape[1],gt_img_save.shape[2],3,3])
    #                     degraded_img_save = np.reshape(np.transpose(degraded_img_save,(1,2,0)),[degraded_img_save.shape[1],degraded_img_save.shape[2],3,3])
                        
    #                     if blur_type == 'anti_ISO':
    #                         _iso_parms = kernel_width_[k]
    #                         name_iso_parm = '{}_{}_{}'.format(_iso_parms[0],_iso_parms[1],_iso_parms[2]*180) ### parms of anti_iso l1_l2_theta
    #                         val_dir = os.path.join(save_path,Mode,'{}-x{}'.format(shuffle_name,sf),'{}-{}-{}-{}'.format(blur_type,noise_type,name_iso_parm,noise_lv))
    #                     elif blur_type == 'Motion':
    #                         _motion_parms = Motion_fraction_[k]### parms of Motion_fraction 
    #                         # name_iso_parm = str('{}_{}_{}'.format(_iso_parms[0],_iso_parms[1],_iso_parms[2]*180)) 
    #                         val_dir = os.path.join(save_path,Mode,'{}-x{}'.format(shuffle_name,sf),'{}-{}-{}-{}'.format(blur_type,noise_type,round(_motion_parms,2),noise_lv))
    #                     if not os.path.exists(val_dir):
    #                         os.makedirs(val_dir)
    #                     dataDict_val = {'img_lr_val':degraded_img_save,'img_hr_val':val_img_save,'kernel_val':current_kernel,'noise_lv':noise_lv}
    #                     save_val_name = os.path.join(val_dir,img_name_+'.pt')
    #                     with open(save_val_name, "wb") as tf:
    #                         pickle.dump(dataDict_val,tf)
            
def main_process(is_shuffle,scale,b_gauss,b_motion,b_unseen,):
    dataset_path = '/mnt/sda/duyou/Project_LFSR/dataset/data_forJDSR/FirstOrder'
    img_path = '/mnt/sda/duyou/Project_LFSR/dataset/ArrayImage/classification_v3/classification_v3'
    
    
    patch_size = 32
    if is_shuffle:
        
        print('start to generating train data for shuffle-x{}'.format(scale))
    else:
        print('start to generating train data for unshuffle-x{}'.format(scale))
            # count  = 0
            
    folder_list = os.listdir(img_path)
    
    for i in range(len(folder_list)):
        print('this file is {} '.format(folder_list[i]))
        file_name_list = []
        name_list = os.listdir(os.path.join(img_path,folder_list[i]))#image names
        for names in range(len(name_list)):
            
            file_name_list.append(os.path.join(img_path,folder_list[i],name_list[names]) ) 
        file_name_list.sort()
        print('this file has {} pairs of image'.format(len(file_name_list)//9))
        print('{} for train,{} for val,{} for test'.format(ceil((len(file_name_list)//9)*0.8),
                                                            ceil((len(file_name_list)//9)*0.9)-ceil((len(file_name_list)//9)*0.8),
                                                            ceil((len(file_name_list)//9))-ceil((len(file_name_list)//9)*0.9)))
        for k in range(len(file_name_list)//9):
            # count += 1
            # print('Making a binary for img : {}'.format(count))
                
            img1 = cv2.imread(file_name_list[k*9+0],cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(file_name_list[k*9+1],cv2.IMREAD_GRAYSCALE)
            img3 = cv2.imread(file_name_list[k*9+2],cv2.IMREAD_GRAYSCALE)
            img4 = cv2.imread(file_name_list[k*9+3],cv2.IMREAD_GRAYSCALE)
            img5 = cv2.imread(file_name_list[k*9+4],cv2.IMREAD_GRAYSCALE)
            img6 = cv2.imread(file_name_list[k*9+5],cv2.IMREAD_GRAYSCALE)
            img7 = cv2.imread(file_name_list[k*9+6],cv2.IMREAD_GRAYSCALE)
            img8 = cv2.imread(file_name_list[k*9+7],cv2.IMREAD_GRAYSCALE)
            img9 = cv2.imread(file_name_list[k*9+8],cv2.IMREAD_GRAYSCALE)
            
            *_,kinds,name = file_name_list[k*9+0].split('/')
            img_num,_ = name.split('_')
            patchidx = kinds+'_'+img_num
            # save_path = os.path.join(dataset_path,patchidx)
            # print(patchidx)
            img_up = np.stack([img1,img2,img3],axis = -1)
            img_center = np.stack([img4,img5,img6],axis = -1)
            img_down = np.stack([img7,img8,img9],axis = -1)
            img_original = np.concatenate([img_up,img_center,img_down],axis = -1) # [h,w,ah,aw] for gray image , [h,w,3,ah,aw]
            # ddd = np.min(img_original/255)
            img_original = img_original/255
            
            if ceil((len(file_name_list)//9)*0.8)>k:
                # pass
                # aaa = aaa
                gen_my_dataset(img_original,patchidx,dataset_path,Mode = 'train',crop_times = 8,sf=scale,shuffle=is_shuffle,cropsize = patch_size,
                               blur_type_=['simga','omsim','unsim'], noise_type_ = ['Gauss','Poisson'],gauss_lv_ = [0,5,10,15,20,25,30],
                               b_gauss = b_gauss,b_motion = b_motion,b_unseen = b_unseen,poisson_size_= [0,200,300,450,500,600,700,1000])
            elif ceil((len(file_name_list)//9)*0.9)<k :
                # pass
                gen_my_dataset(img_original,patchidx,dataset_path,Mode = 'test',sf=scale,shuffle=is_shuffle,blur_type_=['simga','omsim','unsim'],
                               noise_type_ = ['Gauss','Poisson'],gauss_lv_ = [0,5,15,30],poisson_size_= [0,200,450,700],
                               b_gauss = b_gauss,b_motion = b_motion,b_unseen = b_unseen
                               )
            
                
            # else:
            #     # aaa = aaa
            #     # pass
            #     gen_my_dataset(img_original,patchidx,dataset_path,Mode = 'val',shuffle=is_shuffle,cropsize = patch_size,sf=scale,blur_type_=['Motion'],
            #                    noise_type_ = ['Gauss','Poisson'],gauss_lv_ = [0,5,15,30],poisson_size_=[0,200,450,700],
            #                    b_gauss = b_gauss,b_motion = b_motion,b_unseen = b_unseen)



def LFreshape(data, angRes):# stack [B*(C*A*A)*AH*AW] or [B*H*W*A*A] to [B*C*AH*AW]
    dims = data.ndimension()
    if dims ==4:
        B,c, H,W = data.shape#shape B*(C*A*A)*H*W
        
        splited_d = torch.chunk(data,angRes*angRes,axis=1) #groups
        angs_ = []
        for i in range(angRes):
            angs_.append(torch.cat(splited_d[i*angRes:(i+1)*angRes],2))
        lf_rs = torch.cat(angs_[:],-1)
    elif dims ==5:
        B,H,W,ah,aw = data.shape
        if ah!=aw or ah!= angRes or aw!= angRes:
            raise Exception("input angular resolution should meet the default resolution {0},but got {1}".format(angRes,ah))
        else:
            spt_h = torch.chunk(data, ah, 3)
            cat_h = torch.cat(spt_h[:],1)
            spt_w = torch.chunk(cat_h,aw,-1)
            cat_hw = torch.cat(spt_w[:],2)
            out = torch.squeeze(cat_hw,-1)
            lf_rs = out.contiguous().permute(0,3,1,2)
    else:
        raise Exception("input dims should be 4 or 5, but got {}".format(dims))
    
    
    return lf_rs

def LFsplit2d(data, angRes):# split [B*C*AH*AW] to [B*(C*A*A)*H*W]
    H, W = data.shape[-2:]
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[ u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st

if __name__ == '__main__':
    shuffle_options = [True, False]
    scale_options = [2,4]
    
    save_root_k = '/mnt/sda/duyou/Project_LFSR/dataset/data_forJDSR/FirstOrder/kernel'
    save_gauss_k = os.path.join(save_root_k,'gaussK.pt')
    save_motion_k = os.path.join(save_root_k,'motionK.pt')
    save_unseen_k = os.path.join(save_root_k,'unseenK.pt')
    with open(save_gauss_k, 'rb') as _f1:
        gaussK = pickle.load(_f1)
    with open(save_motion_k, 'rb') as _f2:
        motionK = pickle.load(_f2)
    with open(save_unseen_k, 'rb') as _f3:
        unseenK = pickle.load(_f3)
        
    for i in range(len(shuffle_options)):
        shuffle = shuffle_options[i]
        for j in range(len(scale_options)):
            scale = scale_options[j]
            main_process(is_shuffle = shuffle,scale = scale,b_gauss = gaussK,b_motion = motionK,b_unseen = unseenK)
    
    
    ##### get train/test/val images   ######################
    ##### show train/test/val kernels ######################
    # dataset = ['Motion','Gauss']
    # val_img_name = 'car_img50'
    # root_path = '/mnt/sda/duyou/Project_LFSR/dataset/data_forJDSR/FirstOrder/val/unshuffle-x4'
    # save_val_img_path = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/{}'.format(val_img_name),'lr_img')
    # save_val_kernel_path = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/{}'.format(val_img_name),'kernel')
    # if not os.path.exists(save_val_img_path):
    #     os.makedirs(save_val_img_path)
    # if not os.path.exists(save_val_kernel_path):
    #     os.makedirs(save_val_kernel_path)
    
    # kind_list = os.listdir(root_path)
    # ## select names of the searched image 
    # shown_list = []
    # for i in range(len(kind_list)):
    #     if kind_list[i].find(dataset[0])>=0 and kind_list[i].find(dataset[1])>=0:
    #         filename_for_imgs = os.path.join(root_path,kind_list[i],val_img_name+'.pt')
    #         with open(filename_for_imgs, 'rb') as _f:
    #             # print('loading hr image{}'.format(f_hr))
    #             f_imgs = pickle.load(_f)
    #         val_kernel = f_imgs['kernel_val']
    #         val_img = f_imgs['img_lr_val']
    #         # hr_img = f_imgs['img_hr_val']
    #         if kind_list[i].find('Motion-0.2')>=0:
    #             assert kind_list[i]=='G'
    #         save_val_lr_img_name = os.path.join(save_val_img_path,kind_list[i]+'.png')
    #         save_val_kernel_img_name = os.path.join(save_val_kernel_path,kind_list[i]+'.png')
    #         val_img_ = val_img[:,:,0,0]
    #         plt.image.imsave(save_val_lr_img_name,val_img_,cmap='gray')
    #         plt.image.imsave(save_val_kernel_img_name,val_kernel,cmap='gray')
        
    #### show kernel and image in train set
    # train_img_name_ = 'other_img10'
    # root_path = '/mnt/sda/duyou/Project_LFSR/dataset/data_forJDSR/FirstOrder/train/unshuffle-x4'
    # save_train_img_path = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/{}'.format(train_img_name_),'lr_img')
    # save_train_kernel_path = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/{}'.format(train_img_name_),'kernel')
    
    # if not os.path.exists(save_train_img_path):
    #     os.makedirs(save_train_img_path)
    # if not os.path.exists(save_train_kernel_path):
    #     os.makedirs(save_train_kernel_path)
    # for i in range(8):
    #     iter_ = '%02d' % i
    #     train_img_name = os.path.join(root_path,'{}-{}'.format(dataset[0],dataset[1]),train_img_name_+'_{}.pt'.format(iter_))
    #     with open(train_img_name, 'rb') as _f:
    #         # print('loading hr image{}'.format(f_hr))
    #         f_imgs = pickle.load(_f)
    #     train_kernel = f_imgs['kernel_train']
    #     train_lr_img = f_imgs['img_lr_train']
    #     # hr_img = f_imgs['img_hr_train']
        
    #     train_lr_img_name = os.path.join(save_train_img_path,'{}-{}-{}.png'.format(dataset[0],dataset[1],iter_))
    #     train_kernel_img_name = os.path.join(save_train_kernel_path,'{}-{}-{}.png'.format(dataset[0],dataset[1],iter_))
    #     train_lr_img_ = train_lr_img[:,:,0,0]
    #     plt.image.imsave(train_lr_img_name,train_lr_img_,cmap='gray')
    #     plt.image.imsave(train_kernel_img_name,train_kernel,cmap='gray')
        
        
        
    # #### show kernel and image in test set 
    # test_img_name = 'building_img9'
    # root_path = '/mnt/sda/duyou/Project_LFSR/dataset/data_forJDSR/FirstOrder/test/unshuffle-x4'
    # save_test_img_path = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/{}'.format(test_img_name),'lr_img')
    # save_test_kernel_path = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/{}'.format(test_img_name),'kernel')
    # if not os.path.exists(save_test_img_path):
    #     os.makedirs(save_test_img_path)
    # if not os.path.exists(save_test_kernel_path):
    #     os.makedirs(save_test_kernel_path)
        
    # kind_list = os.listdir(root_path)
    # ## select names of the searched image 
    # shown_list = []
    # for i in range(len(kind_list)):
    #     if kind_list[i].find(dataset[0])>=0 and kind_list[i].find(dataset[1])>=0:
    #         filename_for_imgs = os.path.join(root_path,kind_list[i],test_img_name+'.pt')
    #         with open(filename_for_imgs, 'rb') as _f:
    #             # print('loading hr image{}'.format(f_hr))
    #             f_imgs = pickle.load(_f)
    #         test_kernel = f_imgs['kernel_test']
    #         test_lr_img = f_imgs['img_lr_test']
    #         # hr_img = f_imgs['img_hr_val']
    #         # if kind_list[i].find('Motion-0.2')>=0:
    #         #     assert kind_list[i]=='G'
    #         save_test_img_name = os.path.join(save_test_img_path,kind_list[i]+'.png')
    #         save_test_kernel_name = os.path.join(save_test_kernel_path,kind_list[i]+'.png')
    #         test_lr_img_ = test_lr_img[:,:,0,0]
    #         plt.image.imsave(save_test_img_name,test_lr_img_,cmap='gray')
    #         plt.image.imsave(save_test_kernel_name,test_kernel,cmap='gray')
    # '''
    # ### translate .pt HR image into .png
    # '''
    # test_path = '/mnt/sda/duyou/Project_LFSR/dataset/data_forJDSR/FirstOrder/test/HR'
    # save_hr_path = '/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/HR'
    # if not os.path.exists(save_hr_path):
    #     os.makedirs(save_hr_path)
    # imglist = os.listdir(test_path)
    # for i in range(len(imglist)):
        
    #     f_hr_ = os.path.join(test_path,imglist[i])
            
    #     with open(f_hr_, 'rb') as _f1:
    #         # print('getting lr image{}'.format(f_lr))
    #         f_hr = pickle.load(_f1)
    #         # f_hr = np.expand_dims(f_hr,0)
    #     # f_hr = torch.from_numpy(f_hr.astype(np.float32).copy())
    #     f_hr_s = np.reshape(f_hr,[f_hr.shape[0],f_hr.shape[1],-1])
    #     # f_hr_s = LFreshape(f_hr,3)
    #     f_hr_save= f_hr_s[...,4]
    #     f_hr_save = torch.from_numpy(f_hr_save.astype(np.float32).copy())
    #     imgname,_ = imglist[i].split('.')
    #     filename = os.path.join(save_hr_path,imgname+'.png')
        
    #     vutils.save_image(f_hr_save,filename)
    
    
        
        
    # kernel = Motion_blur_kernel_gen(windowSize, kernelSize, isCenter)
    # theta = random.randint(0,360)
    # # kernel = Motion_Blur_kernel(size=kernelSize,theta=theta)
    # psf = PSF(canvas=kernelSize, path_to_save='/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/kernel_motion_v1.png')
    # kernel_motion_ = psf.fit(show=True, save=True)
    # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/kernel_motionv2.png',kernel_motion_[-1],cmap='gray')
    # kernel_motion  =Motion_blur_kernel_gen(windowSize=windowSize,kernelSize=kernelSize,isCenter=isCenter)
    # kernel_disk = Disk_blur_kernel(radis=5)
    # theta=random.random()*np.pi
    # kernel_aniso = anisotropic_Gaussian(ksize=kernelSize, theta=theta)
    # kernel_iso = fspecial('gaussian', kernelSize, kernelSize*random.random())
    # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/kernel_motion.png',kernel_motion,cmap='gray')
    # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/kernel_disk.png',kernel_disk,cmap='gray')
    # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/kernel_aniso.png',kernel_aniso,cmap='gray')
    # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/kernel_iso.png',kernel_iso,cmap='gray')
    # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/kernel_motion.png',kernel_motion,cmap='gray')
    # path = r'/mnt/sda/duyou/Project_LFSR/our_method/AISR_MDSR/src/experiment/ATO/ATOx2b1ep200lr0.0001/results-test/air-conditioning_img9/hr.png'
    # image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    
    # img_ = image/255
    '''
    for j in range (10):
        img_ = image/255
        l1 = random.uniform(0.1,50)
        l2 = random.uniform(0.1,11)
        
        kernelSize = random.choice([5,9,11])
        noise_lv = random.choice([5,15,30])
        blur_type = random.choice(['anti_ISO','ISO','Motion','Disk'])
        degradation = random.choice([0,1,2])
        order = [0,1,2]
        random.shuffle(order)
        print(order)
        for i in range(3):
            
            if order[i]==0:
                img_ = add_noise(img_,Mode='test',noise_level=noise_lv)
            elif order[i]==1:
                img_, kernel = add_blur(img_, Mode = 'test',max_kernel = kernelSize ,min_kernel = 5,blurtype ='anti_ISO',l1=l1,l2=l2 )
            elif order[i] ==2:
                img_ = down_sampling(img_,Mode= 'test',dp_type=degradation, sf=2)
            else:
                raise ValueError('the order : {} is not permitted'.format(order))
            degraded_img = img_
        print(img_.shape)
        print('blur kernel is: {}, with size {}\t noise level is:{}\t degradation is {}'.format(blur_type,kernelSize,noise_lv,degradation))
        plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/degraded_img.png',degraded_img,cmap='gray')
        # l1 = random.uniform(0.1,50)
        # l2 = random.uniform(0.1,11)
        #=========random noise test
        # lv = random.choice([5,15,20])
        # noise_img = add_noise(img_,Mode='test',noise_level=lv)
        # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/noise_img.png',noise_img,cmap='gray')
        #=======random downsampling 
        # dp_type_ = random.choice([0,1,2])
        # img_dp = down_sampling(img_,Mode= 'test',dp_type=dp_type_, sf=2)
        # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/dp_img.png',img_dp,cmap='gray')
        # print(dp_type_)
        #=========random blur kernel 
        # blur_img, kernel = add_blur(image, Mode = 'test',max_kernel = kernelSize ,min_kernel = 5,blurtype ='anti_ISO',l1=l1,l2=l2 )
        # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/blur_img.png',blur_img,cmap='gray')
        # plt.image.imsave(r'/mnt/sda/duyou/Project_LFSR/our_method/LF-JDSR/src/data/kernel_img.png',kernel,cmap='gray')
        # print(kernel.shape)
    '''