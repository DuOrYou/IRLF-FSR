import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    """
    计算一个高斯分布的概率
    :param window_size:
    :param sigma:
    :return:
    """
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / (float(2 * sigma ** 2))) for x in range(window_size)])
    return gauss / gauss.sum()



class KL_loss(torch.nn.Module):
    def __init__(self):
        super(KL_loss, self).__init__()
        self.mse = torch.nn.L1Loss()
        self.criterion = torch.nn.KLDivLoss()

    def forward(self, fea1, fea2):
        b,c,h,w,n = fea1.shape
        loss = 0
        for i in range(n):
            # buffer1 = (fea1[:,:,:,:,i]-torch.min(fea1[:,:,:,:,i]))/(torch.max(fea1[:,:,:,:,i])-torch.min(fea1[:,:,:,:,i])+1e-10)
            # buffer2 = (fea2[:,:,:,:,i]-torch.min(fea2[:,:,:,:,i]))/(torch.max(fea2[:,:,:,:,i])-torch.min(fea2[:,:,:,:,i])+1e-10)
            # G1 = self.gram(buffer1)
            # G2 = self.gram(buffer2)
            # loss += self.mse(G1,G2)
            buffer1 = fea1[:,:,:,:,i]
            buffer2 = fea2[:,:,:,:,i]
            loss += (self.kl_divs(buffer1,buffer2)+self.kl_divs(buffer2,buffer1))/2
        return loss
    def gram(self,x):
        bs, ch, h, w= x.shape
        f = x.contiguous().view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G
    def kl_divs(self,x,y):
        b,c,h,w = x.shape
        
        x = x.contiguous().view(-1, h*w)
        y = y.contiguous().view(-1, h*w)
        
        x = F.log_softmax(x,-1)
        y = F.softmax(y, dim=-1)
        
        klloss_ = self.criterion(x, y)
        return klloss_


