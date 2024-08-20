import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
from einops import rearrange

def gaussian_kernel(k: int, sigma: float):
    ax = torch.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel.to('cuda')
    return kernel / torch.sum(kernel)



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


class GaussionDecomposeBlock(nn.Module):
    def __init__(self,  dimOut) -> None:
        super(GaussionDecomposeBlock,self).__init__()
        self.outdim = dimOut

        # self.dwconv = nn.Conv3d(dimIn,dimOut,(1,ksize,ksize),(1,1,1),(0,ksize//2,ksize//2),groups=dimIn,bias=False)

    def forward(self,x):
        b,c,n,h,w = x.shape
        x_buffer = rearrange(x,'b c n h w -> (b n) c h w')
        short_cut = x_buffer
        low_freq = []
        gaussian = gaussian_kernel(k=5,sigma=1)
        for i in range(self.outdim):
            x_buffer = F.conv2d(x_buffer, gaussian.unsqueeze(0).unsqueeze(0), padding=gaussian.shape[-1]//2)
            low_freq.append(x_buffer)
        low_freq_x = torch.cat(low_freq,1)
        hig_freq_x = short_cut - low_freq_x

        x_lf, x_hf = rearrange(low_freq_x,'(b n) c h w -> b c n h w',n=n),rearrange(hig_freq_x,'(b n) c h w -> b c n h w', n=n)
        return x_lf,x_hf
class freqloss(nn.Module):
    def __init__(self, angRes = 3):
        super(freqloss, self).__init__()
        self.angRes = angRes
        
        self.decompose = GaussionDecomposeBlock(8)
        self.loss_hf = nn.L1Loss()
        self.loss_lf = nn.SmoothL1Loss()

    def forward(self, SR,HR):
        
        # x_hr = LFreshape(HR,self.angRes)
        x_hr = LFsplit(HR, self.angRes)        
        x_hr = rearrange(x_hr,'b (u v) c h w -> b c (u v) h w ',u=self.angRes,v=self.angRes)

        # x_sr = LFreshape(SR,self.angRes)
        x_sr = LFsplit(SR, self.angRes)        
        x_sr = rearrange(x_sr,'b (u v) c h w -> b c (u v) h w ',u=self.angRes,v=self.angRes)

        sr_lf,sr_hf = self.decompose(x_sr)
        hr_lf,hr_hf = self.decompose(x_hr)

        freq_loss = self.loss_hf(sr_hf,hr_hf) + self.loss_lf(sr_lf,hr_lf)


        return freq_loss
