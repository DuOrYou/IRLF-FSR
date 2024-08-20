from fileinput import filename
import os
import math
from decimal import Decimal

from matplotlib.pyplot import show
from torch.utils.tensorboard import SummaryWriter 
import utility
import numpy as np
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import random
import torch.nn.functional as F

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

seed_torch()


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.model_name = args.model
        self.angular = args.angular
        self.ckp = ckp
        
        self.kl_loss = args.KL_loss
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        # self.loader_val = loader.loader_val
        self.decay_every = args.decay_every
        self.model = my_model
        self.loss = my_loss
        self.is_sisr = args.is_sisr
        self.is_misr = args.is_misr
        self.optimizer = utility.make_optimizer(args, self.model)
        self.train_uv = args.train_uv
        self.error_last = 1e8
        self.psnr_temp = 0
        self.dict_ = []
        self.nonuniform_N = args.nonuniform_N
        # if self.nonuniform_N:
        #     unform = 'UN'

        if self.nonuniform_N:
            unform = 'UN'
            if args.is_sisr:
                self.logdir = os.path.join('.', 'experiment','log', args.model,'{}_{}x{}b{}ep{}lr{}uv{}'.format(unform,args.model,args.scale,args.batch_size,args.epochs,args.lr,args.train_uv))
            else:
                self.logdir = os.path.join('.', 'experiment','log', args.model,'{}_{}x{}b{}ep{}lr{}'.format(unform,args.model,args.scale,args.batch_size,args.epochs,args.lr))
        else:

            if args.is_sisr:
                self.logdir = os.path.join('.', 'experiment','log', args.model,'{}x{}b{}ep{}lr{}uv{}'.format(args.model,args.scale,args.batch_size,args.epochs,args.lr,args.train_uv))
            else:
                self.logdir = os.path.join('.', 'experiment','log', args.model,'{}x{}b{}ep{}lr{}'.format(args.model,args.scale,args.batch_size,args.epochs,args.lr))

        # if args.is_sisr:
        #     self.logdir = os.path.join('.', 'experiment','log','{}'.format(args.model),'x{}b{}ep{}lr{}uv{}'.format(args.scale,args.batch_size,args.epochs,args.lr,args.train_uv))
            
        # else:
        #     self.logdir = os.path.join('.', 'experiment','log','{}'.format(args.model),'x{}b{}ep{}lr{}'.format(args.scale,args.batch_size,args.epochs,args.lr))
            
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.summery = SummaryWriter(self.logdir)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
            
            
        # self.ssim_temp = 0
    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        learn_rate = self.optimizer.get_lr()
        if epoch%self.decay_every==0 and epoch!=0:
            learn_rate = learn_rate*0.5
            if self.model_name == 'LFSSR':
                if self.scale ==2 :
                    learn_rate = learn_rate if learn_rate >=1e-6 else 1e-6
                elif self.scale ==2 :
                    learn_rate = learn_rate if learn_rate >=2e-8 else 2e-8
        print('====train====')
        print('epoch{} learn_rate{} '.format(epoch,learn_rate))
        # self.ckp.write_log(
        #     '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(learn_rate))
        # )
        # self.loss.start_log()
        self.model.train()

        timer_model = utility.timer()
        # TEMP
        # self.loader_train.dataset
        
        # devices = []
        # sd = self.model.state_dict()
        # for v in sd.values():
        #     if v.device not in devices:
        #         devices.append(v.device)

        # for d in devices:
        #     print(d)
        # if self.is_sisr:
        #     self.idxU = random.randint(0,self.angular-1)
        #     self.idxV = random.randint(0,self.angular-1)
                # u,v = self.train_uv.split('_')
                # self.idxU = int(u)
                # self.idxV = int(v)
            
        
        for batch, (lr, hr) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            if self.is_sisr:
                
                self.idxU = random.randint(0,self.angular-1)
                self.idxV = random.randint(0,self.angular-1)
                # print('train_u is:{},train_v is:{}'.format(self.idxU,self.idxV))
                
                lr =lr[:,:,:,self.idxU,self.idxV].unsqueeze(1)
                hr =hr[:,:,:,self.idxU,self.idxV].unsqueeze(1)
            # timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            if self.is_misr:
                
                ref_ind = random.randint(0,self.angular*self.angular-1)
                if self.kl_loss:
                    sr,lr_,fea_faf,fea_srf= self.model(lr,ref_ind)
                else:
                    sr,lr_= self.model(lr,ref_ind)
                
            else:
                sr = self.model(lr)
                
            if not self.is_sisr:
                if self.is_misr:
                    b ,h,w,_,_ = hr.shape
                    hr_ = hr.contiguous().view(b,h,w,-1).permute(0,3,1,2)
                    hr = hr_[:,ref_ind,:,:].unsqueeze(1)
                    lr = lr_
                else:
                    hr= utility.LFreshape(hr,self.angular)
                    lr= utility.LFreshape(lr,self.angular)
            # hr3x = utility.LFreshape(hr3x,self.angular)
            # hr4x = utility.LFreshape(hr4x,self.angular)
            # print('lr.shape is {}'.format(lr.shape))
            # print('hr shape is {}'.format(hr.shape))
            if self.kl_loss:
                # print('compute kl_loss')
                loss = self.loss(sr,  hr,fea_faf,fea_srf)
            else:
                loss = self.loss(sr,  hr)
            # print(loss)
            loss.backward()
            # if self.args.gclip > 0:
            #     utils.clip_grad_value_(
            #         self.model.parameters(),
            #         self.args.gclip
            #     )
            # print(loss)
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                print('[{}/{}]\tL1 loss:{:.4f}\t{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    loss,
                    timer_model.release()
                    ))
                # self.summery.add_images('train/sr_images',sr,epoch,dataformats='NCHW')
                # self.summery.add_images('train/hr_images',hr,epoch,dataformats='NCHW')
                # self.summery.add_images('train/lr_images',lr,epoch,dataformats='NCHW')
            #     self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
            #         (batch + 1) * self.args.batch_size,
            #         len(self.loader_train.dataset),
            #         self.loss.display_loss(batch),
            #         timer_model.release(),
            #         timer_data.release()))
                
                # if not batch ==0:
                #     self.test()

            # timer_data.tic()
        # self.summery.add_graph(self.model,lr)
        
        self.summery.add_images('train/sr_images',sr,epoch,dataformats='NCHW')
        self.summery.add_images('train/hr_images',hr,epoch,dataformats='NCHW')
        self.summery.add_images('train/lr_images',lr,epoch,dataformats='NCHW')
        self.summery.add_scalar('train/L1_loss',loss,epoch)
        if self.is_sisr:
            print('train_u is:{},train_v is:{}'.format(self.idxU,self.idxV))
        # self.summery.add_image('images/train/lr',lr,epoch)
        # self.summery.add_image('images/train/sr',sr,epoch)
        # self.summery.add_image('images/train/hr',hr,epoch)
        # self.loss.end_log(len(self.loader_train))
        # self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        self.ckp.save(self, epoch,stage = 'val', is_best=False)

    def val(self):
        torch.set_grad_enabled(False)
        print('====val====')

        epoch = self.optimizer.get_last_epoch()
        # self.ckp.write_log('\nEvaluation:')
        # self.ckp.add_log(
        #     torch.zeros(1, len(self.loader_val),2)
        # )
        self.model.eval()
        psnr = []
        ssim = []
        # timer_val = utility.timer()
        # if self.args.save_results: self.ckp.begin_background()
        # filename = self.test_dir
        
        for idx_data, d in enumerate(self.loader_val):
            lr = d[0]
            hr = d[1]
            
            # hr3x = d[2]
            # hr4x = d[3]
            filename_ = d[2][0]
            # d.dataset
            # for ( lr, hr2x,hr3x,hr4x,filename_) in tqdm(d, ncols=80):
            # filename,_ = filename_.split('.')
            lr, hr= self.prepare(lr, hr)
            if self.is_sisr:
                sr = torch.zeros_like(hr)
                # self.idx = random.randint(0,self.angular*self.angular)
                for u in range(self.angular):
                    for v in range(self.angular):
                        
                        lr_ =lr[:,:,:,u,v].unsqueeze(1)
                        # hr =hr[:,:,:,u,v].unsqueeze(1)
                        sr[:,:,:,u,v] = self.model(lr_).squeeze(1)
                        
                        # psnr_temp = utility.calculate_psnr(sr, hr,is_lf = (not self.is_sisr))
                        # ssim_temp = utility.calcualate_ssim(sr, hr,is_lf = (not self.is_sisr))
                # psnr_
                sr = utility.LFreshape(sr,self.angular)
            else:
                if self.is_misr:
                    
                    b,h,w,_,_ = hr.shape
                    
                    sr_ = hr.contiguous().view(b,h,w,-1).permute(0,3,1,2)
                    sr = torch.zeros_like(sr_)
                    for i in range(self.angular*self.angular):
                        # self.ref_ind = i
                        if self.kl_loss:
                            sr_temp,_,_,_= self.model(lr,i)
                        else:
                            sr_temp,_= self.model(lr,i)
                        sr[:,i,:,:] = sr_temp.squeeze(1)
                    
                    sr = sr.contiguous().permute(0,2,3,1).view(b,h,w,self.angular,self.angular)
                    sr = utility.LFreshape(sr,self.angular)
                else:
                   
                    sr = self.model(lr)
                # if not self.is_sisr:
            hr = utility.LFreshape(hr,self.angular)
            lr= utility.LFreshape(lr,self.angular)
            sr = sr.detach().cpu()
            hr = hr.detach().cpu()
            psnr_ = utility.calculate_psnr(sr, hr,is_lf = (not self.is_sisr))
            ssim_ = utility.calcualate_ssim(sr, hr,is_lf = (not self.is_sisr))
            # hr3x = utility.LFreshape(hr3x,self.angular)
            # hr4x = utility.LFreshape(hr4x,self.angular)
            # save_list = [sr]
            # self.ckp.log[-1, idx_data, 0] = np.mean(np.array(psnr_))
            # self.ckp.log[-1, idx_data, 1] = ssim_
            
            psnr.append(np.mean(np.array(psnr_)))
            ssim.append(ssim_)
            
            # if self.args.save_gt:
            #     save_list.extend([ sr])

            # if self.args.save_results:
            #     self.ckp.save_results(d,filename, save_list)

            # self.ckp.log[-1, idx_data, 0] /= len(d)
            
            # best = self.ckp.log.max(0)
            
        # self.ckp.write_log(
        #     '[PSNR-val \tx{}: {:.2f}]\n[SSIM-val\tx{}:{:.2f}'.format(
        #         self.scale,
        #         self.ckp.log[-1, idx_data, 0],
        #         self.scale,
        #         self.ckp.log[-1, idx_data, 1]
        #         # best[0][idx_data, -1],
        #         # best[1][idx_data].numpy() + 1,
        #         )
        #     )
        
        now_psnr = np.mean(np.array(psnr))
        now_ssim = np.mean(np.array(ssim))
        
        self.summery.add_scalar('psnr/val',now_psnr,epoch)
        self.summery.add_scalar('ssim/val',now_ssim,epoch)
        
        # self.summery.add_images('images/val/lr',lr,epoch,dataformats='NCHW')
        # self.summery.add_images('images/val/sr',sr,epoch,dataformats='NCHW')
        # self.summery.add_images('images/val/hr',hr,epoch,dataformats='NCHW')
        # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_val.toc()))
        # self.ckp.write_log('Saving model...')
        # if epoch%10 == 0:
        #     self.ckp.save(self, epoch,stage = 'val', is_best=False)

        # if not self.args.test_only:
        if epoch!=0 and now_psnr>=self.psnr_temp:
            self.ckp.save(self, epoch,stage = 'val', is_best=True)
        self.psnr_temp = np.mean(np.array(psnr))
        # self.ssim_temp = np.mean(np.array(ssim))
        # self.ckp.write_log(
        #     'Total: {:.2f}s\n'.format(timer_val.toc()), refresh=True
        # )
        
        print('[PSNR-val \tx{}: {:.2f}]\n[SSIM-val\tx{}:{:.2f}]'.format(
                self.scale,
                now_psnr,
                self.scale,
                now_ssim
                )
              )

        torch.set_grad_enabled(True)

    def test(self):
        torch.set_grad_enabled(False)
        print('====test====')
        epoch = self.optimizer.get_last_epoch()
        # self.ckp.write_log('\nTesting:')
        # self.ckp.add_log(
        #     torch.zeros(1, len(self.loader_test),2)
        # )
        self.model.eval()
     
        # timer_test = utility.timer()
        # if self.args.save_results: self.ckp.begin_background()
        # filename = self.test_dir
        
        PSNR_show = []
        SSIM_show = []
        
        for idx_data, d in enumerate(self.loader_test):
            lr = d[0]
            hr = d[1]
            # hr3x = d[2]
            # hr4x = d[3]
            filename_ = d[2][0]
            
            # d.dataset
            # for ( lr, hr2x,hr3x,hr4x,filename_) in tqdm(d, ncols=80):
            if self.args.bytedepth ==16:
                filename = filename_
                lr = utility.LFsplit_b16(lr,self.angular)
                hr = utility.LFsplit_b16(hr,self.angular)
            else:
                
                kinds,imgnum, _ = filename_.split('_')
                filename = kinds+'_'+imgnum
            lr, hr= self.prepare(lr, hr)
            if self.is_sisr:
                sr = torch.zeros_like(hr)
                # self.idx = random.randint(0,self.angular*self.angular)
                for u in range(self.angular):
                    for v in range(self.angular):
                        
                        lr_ =lr[:,:,:,u,v].unsqueeze(1)
                        # print(lr_.shape)
                        # hr =hr[:,:,:,u,v].unsqueeze(1)
                        sr[:,:,:,u,v] = self.model(lr_).squeeze(1)
                        # psnr_temp = utility.calculate_psnr(sr, hr,is_lf = (not self.is_sisr))
                        # ssim_temp = utility.calcualate_ssim(sr, hr,is_lf = (not self.is_sisr))
                # psnr_
                sr = utility.LFreshape(sr,self.angular)
            else:
                if self.is_misr:
                    
                    b,h,w,_,_ = hr.shape
                    
                    sr_ = hr.contiguous().view(b,h,w,-1).permute(0,3,1,2)
                    sr = torch.zeros_like(sr_)
                    
                    for i in range(self.angular*self.angular):
                        self.ref_ind = i
                        if self.kl_loss:
                            sr_temp,_,_,_= self.model(lr,i)
                        else:
                            sr_temp,_= self.model(lr,i)
                        # sr_temp,_= self.model(lr,self.ref_ind)
                        sr[:,i,:,:] = sr_temp.squeeze(1)
                    sr =sr.contiguous().permute(0,2,3,1).view(b,h,w,self.angular,self.angular)
                    
                    sr = utility.LFreshape(sr,self.angular)
                else:
                   
                    sr = self.model(lr)
                # sr = self.model(lr)
            hr = utility.LFreshape(hr,self.angular)
            lr= utility.LFreshape(lr,self.angular)
            hr = torch.squeeze(hr)
            lr = torch.squeeze(lr)
            sr = torch.squeeze(sr)
            # psnr_ = utility.calculate_psnr(sr, hr,is_lf = (not self.is_sisr))
            # ssim_ = utility.calcualate_ssim(sr, hr,is_lf = (not self.is_sisr))
            # hr3x = utility.LFreshape(hr3x,self.angular)
            # hr4x = utility.LFreshape(hr4x,self.angular)
            PSNR,SSIM = utility.cal_metrics(sr, hr, self.angular)
            
            PSNR_show.append(np.mean(np.array(PSNR)))
            SSIM_show.append(np.mean(np.array(SSIM)))
            
            #PSNR, SSIM 排序先行后列
            dict_img = {'img':filename,'psnr':PSNR,'ssim':SSIM} 
            self.dict_.append(dict_img)
    

            save_list = [sr]
            # psnr.append(np.mean(np.array(psnr_)))
            # ssim.append(ssim_)
            # self.ckp.log[-1, idx_data, 1] += utility.calculate_psnr(sr3x, hr3x)
            # self.ckp.log[-1, idx_data, 2] += utility.calculate_psnr(sr4x, hr4x)
            
            # if self.args.save_gt:
            #     save_list.extend([ sr])

            if self.args.save_results:
                self.ckp.save_results(filename, save_list,self.args.bytedepth)
                
        now_psnr = np.mean(np.array(PSNR_show))
        now_ssim = np.mean(np.array(SSIM_show))
        self.summery.add_scalar('psnr/test',now_psnr,epoch)
        self.summery.add_scalar('ssim/test',now_ssim,epoch)
        # self.summery.add_image('images/test/lr',lr,epoch)
        # self.summery.add_image('images/test/sr',sr,epoch)
        # self.summery.add_image('images/test/hr',hr,epoch)
        if not self.args.load:
            # if not args.save:
            #     args.save = now
            if self.args.is_sisr:
                xslxdir_ = os.path.join('.', 'experiment', self.args.model,'{}x{}b{}ep{}lr{}uv{}'.format(self.args.model,self.args.scale,self.args.batch_size,self.args.epochs,self.args.lr,self.args.train_uv))
                xslxdir = os.path.join(xslxdir_,'MTX')
            else:
                xslxdir_ = os.path.join('.', 'experiment', self.args.model,'{}x{}b{}ep{}lr{}'.format(self.args.model,self.args.scale,self.args.batch_size,self.args.epochs,self.args.lr))
                xslxdir = os.path.join(xslxdir_,'MTX')
        if not os.path.exists(xslxdir):
            os.makedirs(xslxdir)
        if self.args.noise_level is not None:
            xslxname = os.path.join(xslxdir,'{}_S-{}.xls'.format(self.args.model, self.args.noise_level[0]))
        else:
            xslxname = os.path.join(xslxdir,'{}_metrics.xls'.format(self.args.model))
        # print(xslxname)
        if self.args.bytedepth==16:
            utility.save_metric_16(xslxname ,self.dict_)
        else:
            
            utility.save_metric_test(xslxname ,self.dict_)

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return False
        else:
            
            epoch = self.optimizer.get_last_epoch() + 1
            if epoch >= self.args.epochs:
                return False
            else:
                
                return epoch








'''

    def test(self):
        torch.set_grad_enabled(False)
        print('====test====')
        epoch = self.optimizer.get_last_epoch()
        print(epoch)
        # self.ckp.write_log('\nTesting:')
        # self.ckp.add_log(
        #     torch.zeros(1, len(self.loader_test),2)
        # )
        self.model.eval()
        psnr = []
        ssim = []
        # timer_test = utility.timer()
        # if self.args.save_results: self.ckp.begin_background()
        # filename = self.test_dir
        
        
        
        for idx_data, d in enumerate(self.loader_test):
            lr = d[0]
            hr = d[1]
            # hr3x = d[2]
            # hr4x = d[3]
            filename_ = d[2][0]
            # d.dataset
            # for ( lr, hr2x,hr3x,hr4x,filename_) in tqdm(d, ncols=80):
            kinds,imgnum, _ = filename_.split('_')
            filename = kinds+'_'+imgnum
            lr, hr= self.prepare(lr, hr)
            if self.is_sisr:
                sr = torch.zeros_like(hr)
                # self.idx = random.randint(0,self.angular*self.angular)
                for u in range(self.angular):
                    for v in range(self.angular):
                        
                        lr_ =lr[:,:,:,u,v].unsqueeze(1)
                        # hr =hr[:,:,:,u,v].unsqueeze(1)
                        sr[:,:,:,u,v] = self.model(lr_).squeeze(1)
                        # psnr_temp = utility.calculate_psnr(sr, hr,is_lf = (not self.is_sisr))
                        # ssim_temp = utility.calcualate_ssim(sr, hr,is_lf = (not self.is_sisr))
                # psnr_
                sr = utility.LFreshape(sr,self.angular)
            else:
                if self.is_misr:
                    
                    b,h,w,_,_ = hr.shape
                    
                    sr_ = hr.contiguous().view(b,h,w,-1).permute(0,3,1,2)
                    sr = torch.zeros_like(sr_)
                    for i in range(self.angular*self.angular):
                        self.ref_ind = i
                        sr_temp,_= self.model(lr,self.ref_ind)
                        sr[:,i,:,:] = sr_temp.squeeze(1)
                    sr =sr.contiguous().permute(0,2,3,1).view(b,h,w,self.angular,self.angular)
                    
                    sr = utility.LFreshape(sr,self.angular)
                else:
                   
                    sr = self.model(lr)
                # sr = self.model(lr)
            hr = utility.LFreshape(hr,self.angular)
            lr= utility.LFreshape(lr,self.angular)
            psnr_ = utility.calculate_psnr(sr, hr,is_lf = (not self.is_sisr))
            ssim_ = utility.calcualate_ssim(sr, hr,is_lf = (not self.is_sisr))
            # hr3x = utility.LFreshape(hr3x,self.angular)
            # hr4x = utility.LFreshape(hr4x,self.angular)
            save_list = [lr,sr,hr]
            psnr.append(np.mean(np.array(psnr_)))
            ssim.append(ssim_)
            # self.ckp.log[-1, idx_data, 1] += utility.calculate_psnr(sr3x, hr3x)
            # self.ckp.log[-1, idx_data, 2] += utility.calculate_psnr(sr4x, hr4x)
            
            # if self.args.save_gt:
            #     save_list.extend([ sr])

            if self.args.save_results:
                self.ckp.save_results(filename, save_list)

            # self.ckp.log[-1, idx_data, 0] /= len(d)
            # print('log matrix shape is {}'.format(self.ckp.log.shape))
            # best = self.ckp.log.max(0)
            # print(self.ckp.log[-1, idx_data,0])
            # aa = best[1][idx_data]
            # print(best[1][idx_data])
            # print('best matrix shape is {}'.format(best.shape))
            # self.ckp.write_log(
            #     '[{}\tx{} PSNR-test:{:.2f}]\n[SSIM-test:{:.2f}'.format(
            #         filename,
            #         self.scale,
            #         self.ckp.log[-1, idx_data, 0],
            #         self.ckp.log[-1, idx_data, 1]
            #         # best[0][idx_data, -1],
            #         # best[1][idx_data].numpy() + 1,
            #         ),show=False
            #     )
        # self.ckp.write_log(
        #         'x{}:[PSNR-show: {:.2f}]\tSSIM-show: {:.2f}]\n'.format(
        #             self.scale,
        #             self.ckp.log[-1, idx_data, 0],
        #             self.ckp.log[-1, idx_data, 1]
        #             # best[0][idx_data, -1],
        #             # best[1][idx_data].numpy() + 1,
        #             )
        #         )
        # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        # self.ckp.write_log('Saving...')

        # if self.args.save_results:
        #     self.ckp.end_background()

        # if not self.args.test_only:
        #     self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        # self.ckp.write_log(
        #     'Total test time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        # )
        
        now_psnr = np.mean(np.array(psnr))
        now_ssim = np.mean(np.array(ssim))
        self.summery.add_scalar('psnr/test',np.mean(np.array(psnr)),epoch)
        self.summery.add_scalar('ssim/test',np.mean(np.array(ssim)),epoch)
        self.summery.add_image('images/test/lr',lr[0,:,:,:],epoch)
        self.summery.add_image('images/test/sr',sr[0,:,:,:],epoch)
        self.summery.add_image('images/test/hr',hr[0,:,:,:],epoch)
        print('[PSNR-test \tx{}: {:.2f}]\n[SSIM-test\tx{}:{:.2f}]'.format(
                self.scale,
                now_psnr,
                self.scale,
                now_ssim
                )
              )
        torch.set_grad_enabled(True)
        


'''
def LFsplit(data, angRes):# split [B*C*AH*AW] to [B*(C*A*A)*H*W]
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st

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
