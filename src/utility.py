import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
from typing import Tuple
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytorch_ssim
import numpy as np
import imageio
from torchvision import utils as vutils
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from skimage import metrics
import xlsxwriter

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

#redefine bg_target
def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get()
            if filename is None: break
            imageio.imwrite(filename, tensor.numpy())
            
class checkpoint():
    
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.scale = args.scale
        self.nonuniform_N = args.nonuniform_N
        self.log = torch.Tensor()
        self.test_unknown = args.test_unknown
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            # if not args.save:
            #     args.save = now
            if self.nonuniform_N:
                unform = 'UN'
                if args.is_sisr:
                    self.dir = os.path.join('.', 'experiment', args.model,'{}_{}x{}b{}ep{}lr{}uv{}'.format(unform,args.model,args.scale,args.batch_size,args.epochs,args.lr,args.train_uv))
                else:
                    self.dir = os.path.join('.', 'experiment', args.model,'{}_{}x{}b{}ep{}lr{}'.format(unform,args.model,args.scale,args.batch_size,args.epochs,args.lr))


            else:

                if args.is_sisr:
                    self.dir = os.path.join('.', 'experiment', args.model,'{}x{}b{}ep{}lr{}uv{}'.format(args.model,args.scale,args.batch_size,args.epochs,args.lr,args.train_uv))
                else:
                    self.dir = os.path.join('.', 'experiment', args.model,'{}x{}b{}ep{}lr{}'.format(args.model,args.scale,args.batch_size,args.epochs,args.lr))
           
        else:
            self.dir = os.path.join('.', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in self.args.data_test:
            
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)
            # self.results_dirs = self.get_path('results-{}'.format(d))

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, stage = '',is_best=False):
        if stage =='train':
            trainer.loss.save(self.dir)
        elif stage=='val':
            # self.plot_psnr(epoch)
            # torch.save(self.log, self.get_path('psnr_log.pt'))
            trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
            
        else:
            raise Exception("stage should be define correctly")
        trainer.optimizer.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)
    def add_log(self, log):
        self.log = torch.cat([self.log, log],axis = 1)

    def write_log(self, log,show = True, refresh=False):
        if show:
            print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        
        fig = plt.figure()
        plt.title('TEST')
        for idx_data, d in enumerate(self.args.data_test):
            
            
            plt.plot(axis,self.log[:, idx_data, 0].numpy(),label='sr2x')
            plt.plot(axis,self.log[:, idx_data, 1].numpy(),label='sr3x')
            plt.plot(axis,self.log[:, idx_data, 2].numpy(),label='sr4x')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_psnr.pdf'))
            plt.close(fig)
    '''    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()'''

    def begin_background(self):
        self.queue = Queue()

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()

    def end_background(self):
        print(self.n_processes)
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, _filename, save_list,bytedepth):
        if self.args.noise_level is not None:
            # print(_filename)
            noise_lv,filename_ = _filename.split('*')
        else:
            filename_ = _filename

        if self.test_unknown:
            filename = self.get_path(
                        'results-unknown',filename_)
            # postfix =  'sr{}x'.format(self.scale)
            postfix =  'lr'
            if not os.path.exists(filename):
                os.makedirs(filename)
            normalized = save_list[0].mul(255 / self.args.rgb_range)
            _tensor=normalized.clone().detach().to(torch.device('cpu'))
                    # _tensor= np.array(_tensor)
            vutils.save_image(_tensor,'{}/{}.png'.format(filename,postfix) )
        else:
            
            if self.args.save_results:

                if bytedepth==16:
                    
                    filename = self.get_path(
                            'results-depth16',filename_)
                else:
                    if self.args.without_gt:
                        filename = self.get_path(
                            'results-real',filename_)
                    else:
                        filename = self.get_path(
                                'results-test',filename_)
            if not os.path.exists(filename):
                    os.makedirs(filename)

            if self.args.noise_level is not None:
                # postfix = ['hr']
                postfix = [ 'sr{}xS-{}'.format(self.scale,noise_lv)]
            else:
                postfix = [ 'sr{}x'.format(self.scale)]
            
            for v, p in zip(save_list, postfix):
                if v[0].dim()<2:
                    normalized = v.mul(255 / self.args.rgb_range)
                else:
                    
                    normalized = v[0].mul(255 / self.args.rgb_range)
                    
                if bytedepth==16:
                    
                    _tensor=(normalized*65535).clone().detach().to(torch.device('cpu'))
                    _tensor= np.array(_tensor).astype(np.uint16)
                    imageio.imsave('{}/{}.png'.format(filename,p) ,_tensor)
                else:
                    # print(normalized.device)
                    aa = normalized.clone().contiguous()
                    bb = aa.detach().cpu()
                    torch.cuda.synchronize()
                    cc= bb.to('cpu')
                    _tensor=aa.detach().to(torch.device('cpu'))
                    # _tensor= np.array(_tensor)
                    # print('save image {}'.format(filename))
                    vutils.save_image(_tensor,'{}/{}.png'.format(filename,p) )
            
            # vutils.save_image(_tensor,'{}/{}.png'.format(filename,p) )
            # tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            # self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calculate_psnr(img1, img2,is_lf  = True):
    # img1 and img2 have range [0, 255]
    
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    psnr = []
    if is_lf:
        
        img1 = LFsplit(img1,3)#B*9*H*W
        img2 = LFsplit(img2,3)
        
        for i in range(img1.shape[1]):
            for j in range(img1.shape[0]):
                
                # h, w = img1.shape[2:]
                img1_ = img1[j,i,:, :]
                img2_ = img2[j,i,:, :]
                img1_ = go2gray(img1_)
                img2_ = go2gray(img2_)
                img1_ = img1_.astype(np.float64)
                img2_ = img2_.astype(np.float64)
                mse_ = np.mean((img1_ - img2_)**2)
                if mse_ == 0:
                    return float('inf')
                psnr_ = 20 * math.log10(255.0 / math.sqrt(mse_))
                psnr.append(psnr_)
    else:
        img1_ = img1
        img2_ = img2
        img1_ = go2gray(img1_)
        img2_ = go2gray(img2_)
        img1_ = img1_.astype(np.float64)
        img2_ = img2_.astype(np.float64)
        mse_ = np.mean((img1_ - img2_)**2)
        if mse_ == 0:
            return float('inf')
        psnr_ = 20 * math.log10(255.0 / math.sqrt(mse_))
        psnr.append(psnr_)
        
                    
    return psnr
    
def save_metric_test(xslxname ,dict):
    # filename = self.get_path('results-Only')
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    # xslxname = filename +'Metric.xls'
    
    output = open(xslxname,'w',encoding='gbk')  #不需要事先创建一个excel表格，会自动生成，gbk为编码方式，支持中文，w代表write
    
    output.write('Name\tnum1\tnum2\tnum3\tnum4\tnum5\tnum6\tnum7\tnum8\tnum9\tP_mean\tnum1\tnum2\tnum3\tnum4\tnum5\tnum6\tnum7\tnum8\tnum9\S_mean\n')
    temp_kind ='.'
    psnr_=0
    ssim_=0
    count=0
    mean_dict = []
    for i in range(len(dict)):
        data = dict[i]
        img_name = data['img']
        kind_ ,_= img_name.split('_')
        
        psnr = data['psnr']
        ssim = data['ssim']
        
        if temp_kind!='.' and kind_!=temp_kind:
            psnr_tem = psnr_/count
            ssim_tem = ssim_/count
            tem_dict =  {'kind':temp_kind,'psnr':psnr_tem,'ssim':ssim_tem} 
            mean_dict.append(tem_dict)
            # output.write('\t')
            # output.write(temp_kind)
            # output.write('\t')
            # output.write(str(psnr_/count))
            
            # output.write('\t')
            # output.write(str(ssim_/count))
            # output.write('\n')     
            psnr_ = np.mean(psnr)
            ssim_ = np.mean(ssim)
            count = 1
        if kind_==temp_kind or temp_kind=='.':
            psnr_ +=np.mean(psnr)
            ssim_ +=np.mean(ssim)
            count +=1
        
        # output.write('\t')
        output.write(img_name)
        for k in range(len(psnr)):
            output.write('\t')
            if isinstance(psnr[k],tuple):
                
                output.write(str(psnr[k][0]))    #write函数不能写int类型的参数，所以使用str()转化
            else:
                output.write(str(psnr[k])) 
        
        # output.write('\n')   #写完一行立马换行
        
        output.write('\t')
        output.write(str(np.mean(psnr)))
        
        for j in range(len(ssim)):
            output.write('\t')
            output.write(str(ssim[j]))
            
        output.write('\t')
        output.write(str(np.mean(ssim)))
        output.write('\n')       #写完一行立马换行
        
        if i ==len(dict)-1:
            # psnr_ +=np.mean(psnr)
            # ssim_ +=np.mean(ssim)
            # count +=1
            psnr_tem = psnr_/count
            ssim_tem = ssim_/count
            tem_dict =  {'kind':temp_kind,'psnr':psnr_tem,'ssim':ssim_tem} 
            mean_dict.append(tem_dict)
        temp_kind,_ = img_name.split('_')
    output.write('\n')
    output.write('\tkinds\tPSNR\tSSIM\n') 
    # output.write('\n')
    for k in range(len(mean_dict)):
        mean_data = mean_dict[k]
        kind_mean_ = mean_data['kind']
        psnr_mean_ = mean_data['psnr']
        ssim_mean_ = mean_data['ssim']
        output.write('\t') 
        output.write(kind_mean_)
        output.write('\t')
        if isinstance(psnr_mean_,tuple):
                
            output.write(str(psnr_mean_[0]))    #write函数不能写int类型的参数，所以使用str()转化
        else:
            output.write(str(psnr_mean_)) 
        # output.write(str(psnr_mean_))
        output.write('\t')
        # output.write(str(ssim_mean_))
        if isinstance(ssim_mean_,tuple):
                
            output.write(str(ssim_mean_[0]))    #write函数不能写int类型的参数，所以使用str()转化
        else:
            output.write(str(ssim_mean_)) 
        output.write('\n')
    output.close()


def save_metric_noise(xslxname ,dict):
    # filename = self.get_path('results-Only')
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    # xslxname = filename +'Metric.xls'
    workbook = xlsxwriter.Workbook(xslxname)
    headers = ['Name', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9','AvgP', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9','AvgS']
    split_dict = {}
    sum_sheet = workbook.add_worksheet('SUM')
    
    for sample in dict:
        kind = sample['kind']
        if kind not in split_dict:
            split_dict[kind] = []
        split_dict[kind].append(sample)
    # kind_list = split_dict.keys()
    noise_count = 0
    for temp_kind in split_dict.keys():
        # temp_kind = kind_list[i]
        class_sheet = workbook.add_worksheet(temp_kind)
        write_dict = split_dict[temp_kind]

        for col in range(len(headers)):
            class_sheet.write(0, col, headers[col])

        sum_psnr = 0
        sum_ssim = 0
        for row in range(len(write_dict)):

            write_row = row +1
            temp_data = write_dict[row]
            temp_name = temp_data['img']
            *_,temp_name = temp_name.split('*')
            psnr_of_allview = temp_data['psnr']
            ssim_of_allview = temp_data['ssim']
            data_list = []
            data_list.append(temp_name)
            for k in range(len(psnr_of_allview)):
                if math.isnan(psnr_of_allview[k]) :
                    data_list.append(0)
                else:
                    data_list.append(psnr_of_allview[k])
            if math.isnan(np.mean(psnr_of_allview)):
                data_list.append(0)
                sum_psnr += 0
            else:
                data_list.append(np.mean(psnr_of_allview))
                sum_psnr += np.mean(psnr_of_allview)

            for k in range(len(ssim_of_allview)):
                if math.isnan(ssim_of_allview[k]) :
                    data_list.append(0)
                else:
                    data_list.append(ssim_of_allview[k])

            if math.isnan(np.mean(ssim_of_allview)):
                data_list.append(0)
                sum_ssim += 0
            else:
                data_list.append(np.mean(ssim_of_allview))
                sum_ssim += np.mean(ssim_of_allview)
                


            for len_of_datalist in range(len(data_list)):
                
                class_sheet.write(write_row, len_of_datalist, data_list[len_of_datalist])

        MeanPsnrofOneSheet = round(sum_psnr / len(write_dict),2)
        MeanSsimofOneSheet = round(sum_ssim / len(write_dict),4)

        sum_metric = str(MeanPsnrofOneSheet) + '/' + str(MeanSsimofOneSheet)
        noise_count += 1 
        sum_sheet.write(1, noise_count, sum_metric)
    workbook.close()


def calculate_average(data):
    averages = []
    for col in range(2, 11):  # 从第3列开始（索引为2）是成绩列
        total = sum(row[col] for row in data)
        average = total / len(data)
        averages.append(average)
    return averages

def save_metric_16(xslxname ,dict):
    # filename = self.get_path('results-Only')
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    # xslxname = filename +'Metric.xls'
    
    output = open(xslxname,'w',encoding='gbk')  #不需要事先创建一个excel表格，会自动生成，gbk为编码方式，支持中文，w代表write
    
    output.write('Name\tnum1\tnum2\tnum3\tnum4\tnum5\tnum6\tnum7\tnum8\tnum9\tP_mean\tnum1\tnum2\tnum3\tnum4\tnum5\tnum6\tnum7\tnum8\tnum9\S_mean\n')
    temp_kind ='.'
    psnr_=0
    ssim_=0
    count=0
    mean_dict = []
    for i in range(len(dict)):
        data = dict[i]
        img_name = data['img']
        # kind_ ,_= img_name.split('_')
        
        psnr = data['psnr']
        ssim = data['ssim']
        
        # if temp_kind!='.' and kind_!=temp_kind:
        #     psnr_tem = psnr_/count
        #     ssim_tem = ssim_/count
        #     tem_dict =  {'kind':temp_kind,'psnr':psnr_tem,'ssim':ssim_tem} 
        #     mean_dict.append(tem_dict)
        #     # output.write('\t')
        #     # output.write(temp_kind)
        #     # output.write('\t')
        #     # output.write(str(psnr_/count))
            
        #     # output.write('\t')
        #     # output.write(str(ssim_/count))
        #     # output.write('\n')     
        #     psnr_ = np.mean(psnr)
        #     ssim_ = np.mean(ssim)
        #     count = 1
        # if kind_==temp_kind or temp_kind=='.':
        #     psnr_ +=np.mean(psnr)
        #     ssim_ +=np.mean(ssim)
        #     count +=1
        
        # output.write('\t')
        output.write(img_name)
        for k in range(len(psnr)):
            output.write('\t')
            if isinstance(psnr[k],tuple):
                
                output.write(str(psnr[k][0]))    #write函数不能写int类型的参数，所以使用str()转化
            else:
                output.write(str(psnr[k])) 
        
        # output.write('\n')   #写完一行立马换行
        
        output.write('\t')
        output.write(str(np.mean(psnr)))
        
        for j in range(len(ssim)):
            output.write('\t')
            output.write(str(ssim[j]))
            
        output.write('\t')
        output.write(str(np.mean(ssim)))
        output.write('\n')       #写完一行立马换行
        
        # if i ==len(dict)-1:
        #     # psnr_ +=np.mean(psnr)
        #     # ssim_ +=np.mean(ssim)
        #     # count +=1
        #     psnr_tem = psnr_/count
        #     ssim_tem = ssim_/count
        #     tem_dict =  {'kind':temp_kind,'psnr':psnr_tem,'ssim':ssim_tem} 
        #     mean_dict.append(tem_dict)
        # temp_kind,_ = img_name.split('_')
    # output.write('\n')
    # output.write('\tkinds\tPSNR\tSSIM\n') 
    # output.write('\n')
    # for k in range(len(mean_dict)):
    #     mean_data = mean_dict[k]
    #     kind_mean_ = mean_data['kind']
    #     psnr_mean_ = mean_data['psnr']
    #     ssim_mean_ = mean_data['ssim']
    #     output.write('\t') 
    #     output.write(kind_mean_)
    #     output.write('\t')
    #     if isinstance(psnr_mean_,tuple):
                
    #         output.write(str(psnr_mean_[0]))    #write函数不能写int类型的参数，所以使用str()转化
    #     else:
    #         output.write(str(psnr_mean_)) 
    #     # output.write(str(psnr_mean_))
    #     output.write('\t')
    #     # output.write(str(ssim_mean_))
    #     if isinstance(ssim_mean_,tuple):
                
    #         output.write(str(ssim_mean_[0]))    #write函数不能写int类型的参数，所以使用str()转化
    #     else:
    #         output.write(str(ssim_mean_)) 
    #     output.write('\n')
    output.close()



    
def calcualate_ssim(img1,img2,is_lf = True):
    
    if is_lf:
        img1 = LFsplit(img1,3)#B*9*H*W
        img2 = LFsplit(img2,3)
    # else:
        
    # img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
    # img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0   
    # img1 = Variable( img1,  requires_grad=False)    # torch.Size([256, 256, 3])
    # img2 = Variable( img2, requires_grad = False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value

def avgGradient(image):
    # image = LFsplit(image,3)
    # image = image.cpu().numpy()
    
    imageAG = []
    for u in range(image.shape[0]):
        for v in range(image.shape[1]):
            image_ = image[u,v,:,:]
            image_ = go2gray(image_)
            # image_ = image_.cpu().numpy()
            tmp = 0
            for i in range(image_.shape[0]-1):
                for j in range(image_.shape[1]-1):
                    dx = image_[i,j+1]-image_[i,j]
                    dy = image_[i+1,j]-image_[i,j]
                    ds = math.sqrt((dx*dx+dy*dy)/2)
                    tmp += ds
    
            imageAG.append(tmp/((image_.shape[1]-1)*(image_.shape[0]-1)))
    return imageAG

def spatialF(image):
    image = LFsplit(image,3)
    # image = image.cpu().numpy()
    SF = []
    for u in range(image.shape[0]):
        for v in range(image.shape[1]):
            image_ = image[u,v,:,:]
            image_ = go2gray(image_)
            # image_ = image_.cpu().numpy()
            M = image_.shape[0]
            N = image_.shape[1]
            cf = 0
            rf = 0
            for i in range(1,M-1):
                for j in range(1,N-1):
                    dx = float(image_[i,j-1])-float(image_[i,j])
                    rf += dx**2
                    dy = float(image_[i-1,j])-float(image_[i,j])
                    cf += dy**2

            RF = math.sqrt(rf/(M*N))
            CF = math.sqrt(cf/(M*N))
            SF.append(math.sqrt(RF**2+CF**2))
    return SF

def standardD(image):
    image = LFsplit(image,3)
    # image = image.cpu().numpy()
    SD = []
    for u in range (image.shape[0]):
        for v in range(image.shape[1]):
            image_ = image[u,v,:,:]
            image_ = go2gray(image_)
            # image_ = image_.cpu().numpy()
            std = np.std(image_, ddof = 1)
            SD.append(std)
    return SD
def go2gray(x):
    # print(np.any(x.cpu().numpy() == 0))
    # x_ = x.cpu().numpy()
    if torch.is_tensor(x):
        
        x_max = torch.max(x)
        x_min = torch.min(x)
        x_1 = (x-x_min)/(x_max-x_min)
        # print(x_1.device)
        X_norm = (x_1*255).detach().cpu().numpy()
    else:
        x_max = np.max(x)
        x_min = np.min(x)
        x_1 = (x-x_min)/(x_max-x_min)
        X_norm = (x_1*255).round()
    # x_nmax = np.max(X_norm)
    # x_nmin = np.min(X_norm)
    # X_scaled = X_std * (max - min) + min
    return  X_norm
def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0
    
    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

# redefine lambda x: x.requires_grad
def return_x(x):
    return x.requires_grad

#redefine lambda x: int(x)
def return_y(y):
    return int(y)

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(return_x, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(return_y, args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

# def data_norm(X):
#     x_ = (X-127.5)/127.5
#     return  x_

def LFsplit(data, angRes):# split [B*C*AH*AW] to [B*(C*A*A)*H*W]
    b, _,H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.cat(data_sv, dim=1)
    
    return data_st

def LFsplit_b16(data, angRes):# split [B*C*AH*AW] to [B*(C*A*A)*H*W]
    b, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:,  u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=-1)
    # print(data_st.shape)
    data_st = data_st.contiguous().view(-1,h,w,angRes,angRes)
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

def cal_psnr_(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()
    # print(img1_np.max())
    # print(img2_np.max())
    # print(img2_np.min())
    # print(img1_np.min())
    return metrics.peak_signal_noise_ratio(img1_np, img2_np, data_range=1)

def cal_ssim_(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True,data_range =1)

def cal_metrics(img1, img2, angRes):
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)

    [U, V, h, w] = img1.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')

    for u in range(U):
        for v in range(V):
            PSNR[u, v] = cal_psnr_(img1[u, v, :, :], img2[u, v, :, :])
            SSIM[u, v] = cal_ssim_(img1[u, v, :, :], img2[u, v, :, :])
            pass
        pass
    PSNR = np.reshape(PSNR,-1)
    SSIM = np.reshape(SSIM,-1)
    # psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    # ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR, SSIM

    