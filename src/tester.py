from fileinput import filename
import os
import math
from decimal import Decimal
import pickle
from einops import rearrange
from cv2 import mean
import utility
import numpy as np
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from torchvision import utils as vutils
import torch.nn.functional as F
class Tester():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.model_name = args.model
        self.angular = args.angular
        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = my_model
        self.is_misr = args.is_misr
        self.is_sisr = args.is_sisr
        self.kl_loss = args.KL_loss
        self.train_uv = args.train_uv
        self.bytedepth = args.bytedepth
        self.nonuniform_N = args.nonuniform_N
        self.dir = os.path.join('.', 'experiment', args.model)
        # self.temp_noise_lv = args.noise_level[3]
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        if self.nonuniform_N:
            unform = 'UN'
            self.methods = '{}_{}x{}b{}ep{}lr{}'.format(unform,args.model,args.scale,args.batch_size,args.epochs,args.lr)
        else:

            self.methods = '{}x{}b{}ep{}lr{}'.format(args.model,args.scale,args.batch_size,args.epochs,args.lr)

        self.dict_ = []

    def test(self):
        torch.set_grad_enabled(False)
        print('====test====')
        # self.ckp.write_log('\nTesting:')
        # self.ckp.add_log(
        #     torch.zeros(1, len(self.loader_test),2)
        # )
        self.model.eval()
     
        # timer_test = utility.timer()
        # if self.args.save_results: self.ckp.begin_background()
        # filename = self.test_dir
        
        
        
        for idx_data, d in enumerate(self.loader_test):
            
            lr = d[0]
            hr = d[1]
            # hr3x = d[2]
            # hr4x = d[3]
            filename_ = d[2][0]
            print(filename_)
            # d.dataset
            # for ( lr, hr2x,hr3x,hr4x,filename_) in tqdm(d, ncols=80):
            if self.bytedepth ==16:
                filename = filename_
                lr = utility.LFsplit_b16(lr,self.angular)
                hr = utility.LFsplit_b16(hr,self.angular)
            else:
                if self.args.without_gt:
                    filename =filename_
                    # filename,_ = filename_.split('_')
                else:
                    if self.args.noise_level is not None:
                        noiselv,_imagenames = filename_.split('*')
                        kinds,imgnum, _ = _imagenames.split('_')
                        filename =noiselv + '*' + kinds+'_'+imgnum
                    else:
                        kinds,imgnum, _ = filename_.split('_')
                        filename = kinds+'_'+imgnum
            lr, hr= self.prepare(lr, hr)

            if not self.args.test_via_patches:
                sr = self.GetOutputImgs(lr)
            else:
                ## lr >> h w u v
                minibatch_for_test = 32
                patchsize_for_test = 32
                stride_for_test = 16
                subLFin = LFdivide(lr,self.angular,patch_size=patchsize_for_test,stride=stride_for_test)
                

                numU, numV = subLFin.shape[0],subLFin.shape[1]
                subLFin = rearrange(subLFin, 'n1 n2 h w a1 a2-> (n1 n2) h w a1 a2')
                # sub_info =rearrange(sub_info, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
                subLFout = torch.zeros(numU * numV, 1, self.angular* patchsize_for_test * self.scale,
                                    self.angular*patchsize_for_test * self.scale)
                for i in range(0, numU * numV, minibatch_for_test):
                    print(i)
                    tmp = subLFin[i:min(i + minibatch_for_test, numU * numV), ...]
                    out = self.GetOutputImgs(tmp)
                    subLFout[i:min(i + minibatch_for_test, numU * numV), ...] = out

                subLFout = rearrange(subLFout, '(n1 n2) 1 (a1 h) (a2 w)-> n1 n2 a1 a2 h w', n1=numU, n2=numV,a1=self.angular,a2=self.angular)

                ''' Restore the Patches to LFs '''
                sr = LFintegrate(subLFout, self.angular, patchsize_for_test * self.scale,
                                   stride_for_test * self.scale, hr.shape[1], hr.shape[2])
                sr = rearrange(sr,'u v h w -> 1 1 (u h) (v w)')
                
            
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
            print(self.args.without_gt)
            if not self.args.without_gt:

                PSNR,SSIM = utility.cal_metrics(sr, hr, self.angular)
                if self.args.noise_level is not None:                 
                    #PSNR, SSIM 排序先行后列
                    dict_img = {'img':filename,'psnr':PSNR,'ssim':SSIM,'kind': noiselv} 
                    self.dict_.append(dict_img)
                else:
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
                print(filename)
                self.ckp.save_results(filename, save_list,self.bytedepth)
        if self.bytedepth==16:
            xslxdir = '/mnt/sda/duyou/Project_LFSR/our_method/AISR_MDSR/src/experiment/Metrics_16'
        else:
            xslxdir = '/mnt/sda/duyou/Project_LFSR/our_method/AISR_MDSR/src/experiment/NewMetrics'

        # if self.nonuniform_N:
        #     xslxdir = os.path.join(xslxdir,'UniformNMetrics')
        if not os.path.exists(xslxdir):
            os.makedirs(xslxdir)

        if self.args.noise_level is not None:
            xslxname = os.path.join(xslxdir,'{}_S.xlsx'.format(self.methods))
        else:
            xslxname = os.path.join(xslxdir,'{}_metrics.xlsx'.format(self.methods))

        # xslxname = os.path.join(xslxdir,'{}_metrics.xls'.format(self.methods))
        if not self.args.without_gt:
            if self.bytedepth==16:
                utility.save_metric_16(xslxname ,self.dict_)
            else:
                if self.args.noise_level is not None:
                     utility.save_metric_noise(xslxname ,self.dict_) 
                else:

                    utility.save_metric_test(xslxname ,self.dict_)

    def GetOutputImgs(self,lr):
        b,h,w,u,v = lr.shape
        sr_temp = torch.zeros_like(lr).repeat(1,self.scale,self.scale,1,1)
        if self.is_sisr:
            lr_ = rearrange(lr,'b h w u v -> (b u v) 1 h w') 
            sr_temp_sisr = rearrange(sr_temp,'b h w u v -> (b u v) 1 h w')                           
            # for u in range(self.angular):
            #     for v in range(self.angular):                    
            #         lr_ =lr[:,:,:,u,v].unsqueeze(1)
                    # print(lr_.shape)
                    # hr =hr[:,:,:,u,v].unsqueeze(1)
            sr_temp_sisr = self.model(lr_)
                    # psnr_temp = utility.calculate_psnr(sr, hr,is_lf = (not self.is_sisr))
                    # ssim_temp = utility.calcualate_ssim(sr, hr,is_lf = (not self.is_sisr))
            # psnr_
            sr_out = rearrange(sr_temp_sisr,'(b u v) 1 h w -> b 1 (u h) (v w)',u=self.angular,v=self.angular)
        else:
            if self.is_misr:
                sr_temp_buffer = rearrange(sr_temp,'b h w u v -> b (u v) h w')
                for i in range(self.angular*self.angular):
                    self.ref_ind = i
                    if self.kl_loss:
                        sr_temp_buffer,_,_,_= self.model(lr,i)
                    else:                                
                            
                        sr_temp_buffer[:,i:i+1,...],_ = self.model(lr,i)

                sr_out = rearrange(sr_temp_buffer,'b (u v) h w -> b 1 (u h) (v w)',u=self.angular,v=self.angular)
            else:
                sr_out = self.model(lr)
        return sr_out
        

    # def test(self):
    #     torch.set_grad_enabled(False)

    #     # self.ckp.write_log('\nTesting:')
        
    #     self.model.eval()

    #     timer_test = utility.timer()
    #     # if self.args.save_results: self.ckp.begin_background()
    #     # filename = self.test_dir
    #             # d.dataset
        
    #     # self.ckp.add_log(
    #     #     torch.zeros(1, len(self.loader_test_only),9)
    #     #     )
    #     name_list = []
    #     AG_list = []
    #     SD_list = []
    #     SF_list = []
    #     timelist = []
        
    #     # print(self.args.test_only_dir)
    #     for idx_data, d in enumerate(self.loader_test_only):
    #         # print('begin test only')
    #         lr_ = d[0]
    #         # print(type(lr_))
    #         filename = d[1][0]
    #         lr = lr_.to(torch.device('cuda'))
            
    #         timer_forward = utility.timer()
    #         print('predict img {}'.format(filename))
    #         sr = self.model(lr)
    #         timelist.append(timer_forward.toc())
    #         # print('predict time is {}'.format(timer_forward.toc()))
            
    #         # print(self.model.state_dict()['model.m_isb_ang1.d_h1.weight'][1,0,:,:])
    #         # img_list = [sr, f_spa[0,0,:,:],f_ang[0,0,:,:],fs_srb1[0,0,:,:],fs_isb1[0,0,:,:],fa_isb1[0,0,:,:],fa_srb1[0,0,:,:],fs_conv[0,0,:,:],fs_[0,0,:,:],fa_conv[0,0,:,:],f_as[0,0,:,:]]
            
    #         self.save_imgs(filename, sr)
    #         name_list.append(filename)
            
            
    #         # self.ckp.write_log(
    #         #     'predict time: {:.2f}s\n'.format(timer_forward.toc()), refresh=True)
            
    #         #compute metrics
    #         AGtimer = utility.timer()
    #         AG = np.mean(utility.avgGradient(sr))
    #         print('AG is:{:.4f}, compute time:{:.4f}'.format(AG,AGtimer.toc()))
    #         AG_list.append(AG)
            
    #         # SDtimer = utility.timer()
    #         SD = np.mean(utility.standardD(sr)) 
    #         # print('SD is:{:.4f}, compute time:{:.4f}'.format(SD,SDtimer.toc()))
    #         SD_list.append(SD)
            
    #         # SFtimer = utility.timer()
    #         SF = np.mean(utility.spatialF(sr))
            
    #         # print('SF is:{:.4f}, compute time:{:.4f}'.format(SF,SFtimer.toc()))
    #         SF_list.append(SF)
            
    #     self.save_metric(name_list,AG_list,SD_list,SF_list,timelist)
            
            # #save results
            # lr = utility.LFreshape(lr,3)
            # save_list = [sr,lr]
            # self.save_results(filename, save_list)
            
            # self.ckp.write_log(
            #     '[metric-test x{}\tAG:{:.2f}\tSD:{:.2f}\tSF:{:.2f}]'.format(
            #     self.scale,
            #     np.mean(np.array(AG)),
            #     np.mean(np.array(SD)),
            #     np.mean(np.array(SF))
            #     )
            # )
            
            # self.ckp.write_log('======done Test=====')

    def save_metric(self,imglist,AGlist,SDlist,SFlist,timelist):
        filename = self.get_path('results-Only')
        if not os.path.exists(filename):
            os.makedirs(filename)
        xls_name = filename +'Metric.xls'
        
        output = open(xls_name,'w',encoding='gbk')  #不需要事先创建一个excel表格，会自动生成，gbk为编码方式，支持中文，w代表write
        output.write('Name\tAG\tSD\tSF\ttime\n')
        for i in range(len(imglist)):
            
            output.write(str(imglist[i]))    #write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')   #相当于Tab一下，换一个单元格
            output.write(str(AGlist[i]))
            output.write('\t')
            output.write(str(SDlist[i]))
            output.write('\t')
            output.write(str(SFlist[i]))
            output.write('\t')
            output.write(str(timelist[i]))
            output.write('\n')       #写完一行立马换行
        output.close()


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]
    
    def save_results(self, _filename, save_list):
        
        filename = self.get_path(
                'results-Only',_filename)
        if not os.path.exists(filename):
                os.makedirs(filename)
        postfix = ( 'sr{}x'.format(self.scale),'lr')
        for v, p in zip(save_list, postfix):
            
            normalized = v[0].mul(255 / self.args.rgb_range)
            _tensor=normalized.clone().detach().to(torch.device('cpu'))
            vutils.save_image(_tensor,'{}/{}.png'.format(filename,p) )
    def save_imgs(self, _filename, save_tensor):
        
        filename = self.get_path(
                'results-Only',_filename)
        if not os.path.exists(filename):
                os.makedirs(filename)
        # postfix = ('_sr', 'f_spa','f_ang','fs_srb1','fs_isb1','fa_isb1','fa_srb1','fs_conv','fs_','fa_conv','f_as')
        # for v, p in zip(save_list, postfix):
            
        normalized = utility.go2gray(save_tensor.squeeze())
        normalized_ = torch.from_numpy(normalized).mul(1 / 255)
        _tensor=normalized_.clone().detach().to(torch.device('cpu'))
        vutils.save_image(_tensor,'{}/sr_{}x.png'.format(filename,self.scale) )
    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

   
def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '1 h w a1 a2 -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    # pad = torch.nn.ReflectionPad2d(padding=(bdr, bdr+stride-1, bdr, bdr+stride-1))
    # data_pad = pad(data)
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 h w a1 a2',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF

def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out

def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]


    return outLF