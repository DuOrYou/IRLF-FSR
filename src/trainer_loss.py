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


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.angular = args.angular
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loader_val = loader.loader_val
        self.model = my_model
        self.loss = my_loss
        self.gpus = args.n_GPUs
        self.optimizer = utility.make_optimizer(args, self.model)
        self.logdir = os.path.join('.', 'experiment','log', args.model)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.summery = SummaryWriter(self.logdir)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.psnr_temp = 0
        # self.ssim_temp = 0
    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        learn_rate = self.optimizer.get_lr()
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
        for batch, (lr, hr) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            # timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr,aln_loss= self.model(lr)
            if self.gpus >1:
                aln_loss = torch.mean(aln_loss)
            hr= utility.LFreshape(hr,self.angular)
            # hr3x = utility.LFreshape(hr3x,self.angular)
            # hr4x = utility.LFreshape(hr4x,self.angular)
            # print(aln_loss)
            loss = self.loss(sr,  hr)+aln_loss
            
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
            #     self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
            #         (batch + 1) * self.args.batch_size,
            #         len(self.loader_train.dataset),
            #         self.loss.display_loss(batch),
            #         timer_model.release(),
            #         timer_data.release()))
                
                # if not batch ==0:
                #     self.test()

            # timer_data.tic()
            
        self.summery.add_scalar('train/L1_loss',loss,epoch)
        # self.summery.add_image('images/train/lr',lr,epoch)
        # self.summery.add_image('images/train/sr',sr,epoch)
        # self.summery.add_image('images/train/hr',hr,epoch)
        # self.loss.end_log(len(self.loader_train))
        # self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

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
            sr,_ = self.model(lr)
            
            hr = utility.LFreshape(hr,self.angular)
            lr= utility.LFreshape(lr,self.angular)
            psnr_ = utility.calculate_psnr(sr, hr)
            ssim_ = utility.calcualate_ssim(sr, hr)
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
        self.summery.add_scalar('psnr/val',np.mean(np.array(psnr)),epoch)
        self.summery.add_scalar('ssim/val',np.mean(np.array(ssim)),epoch)
        self.summery.add_images('images/val/lr',lr,epoch,dataformats='NCHW')
        self.summery.add_images('images/val/sr',sr,epoch,dataformats='NCHW')
        self.summery.add_images('images/val/hr',hr,epoch,dataformats='NCHW')
        # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_val.toc()))
        # self.ckp.write_log('Saving model...')
        if epoch%10 == 0:
            self.ckp.save(self, epoch,stage = 'val', is_best=False)

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
            filename,_ = filename_.split('.')
            lr, hr= self.prepare(lr, hr)
            # print('weights before test is {}'.format(self.model.state_dict()['model.m_isb_ang1.d_h1.weight'][1,0,:,:]))
            sr,_ = self.model(lr)
            # print('weights after test is {}'.format(self.model.state_dict()['model.m_isb_ang1.d_h1.weight'][1,0,:,:]))
            hr = utility.LFreshape(hr,self.angular)
            lr= utility.LFreshape(lr,self.angular)
            psnr_ = utility.calculate_psnr(sr, hr)
            ssim_ = utility.calcualate_ssim(sr, hr)
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

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    # def terminate(self):
    #     if self.args.test_only:
    #         self.test()
    #         return True
    #     else:
    #         epoch = self.optimizer.get_last_epoch() + 1
    #         return epoch >= self.args.epochs
