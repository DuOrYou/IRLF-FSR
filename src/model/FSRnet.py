import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
# from utils.utils import LF_rgb2ycbcr, LF_ycbcr2rgb, LF_interpolate
from einops import rearrange
import time
def make_model(args, parent=False):
    return Net(args)
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        channels = args.n_feats

        self.angRes = args.angular
        self.scale = args.scale
        ksize = 5
        self.groups = 5

        #################### Initial Feature Extraction #####################
        self.to_frequency = FrequencyDecomposeBlock(1,channels,ksize=ksize)
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        self.altblock = nn.Sequential(*[CascadeTransformerBlock(channels=channels,angRes=self.angRes,ksize=5) for i in range(self.groups)])

        ########################### UP-Sampling #############################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels*2, channels * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, lr, info=None):
       
        # x = LFreshape(lr,self.angRes)
        x_multi = rearrange(lr,'b h w u v -> b (u v) 1 h w')
        # LFsplit(x, self.angRes)
        # rgb2ycbcr
        # lr_ycbcr = LF_rgb2ycbcr(lr)
        b, n, c, h, w = x_multi.shape
        
        x_multi_ = rearrange(x_multi,'b (u v) c h w -> (b u v) c h w',u=self.angRes,v=self.angRes)
        
        x_upscale = F.interpolate(x_multi_, scale_factor=self.scale, mode='bicubic', align_corners=False)
        sr_ycbcr = rearrange(x_upscale,'(b u v) c h w -> b (u v) c h w', u=self.angRes, v=self.angRes)
        # Initial Feature Extraction
        x_ = rearrange(x_multi, 'b (u v) 1 h w-> b 1 (u v) h w', u=self.angRes, v=self.angRes)
        t_1 = time.time()
        print(x_.shape)
        x_lf,x_hf = self.to_frequency(x_)
        t_2 = time.time()
        buffer_lf,buffer_hf = self.conv_init(x_lf), self.conv_init(x_hf)
        t_3 = time.time()

        # Deep Spatial-Angular Correlation Learning
        buffer_lf_,buffer_hf_ = self.altblock((buffer_lf,buffer_hf))
        t_4 = time.time()
        buffer_lf = buffer_lf_ + buffer_lf
        buffer_hf = buffer_hf_ + buffer_hf
        # UP-Sampling
        buffer = rearrange(torch.cat([buffer_lf, buffer_hf], 1), 'b c (u v) h w -> (b u v) c h w', u=self.angRes, v=self.angRes)
        out = self.upsampling(buffer) 
        t_5 = time.time()
        buffer = rearrange(out,'(b u v) c h w -> b (u v) c h w',u=self.angRes,v=self.angRes)
        out = FormOutput(buffer)+ FormOutput(sr_ycbcr)
        print('decompose time: {}'.format(t_2-t_1))
        print('conv_init time: {}'.format(t_3-t_2))
        print('altblock time: {}'.format(t_4-t_3))
        print('upsampling time: {}'.format(t_5-t_4))
        # y = rearrange(y, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)

        # ycbcr2rgb
        return out

class FrequencyDecomposeBlock(nn.Module):
    def __init__(self, dimIn, dimOut,ksize = 5) -> None:
        super(FrequencyDecomposeBlock,self).__init__()

        self.dwconv = nn.Conv3d(dimIn,dimOut,(1,ksize,ksize),(1,1,1),(0,ksize//2,ksize//2),groups=dimIn,bias=False)

    def forward(self,x):
        b,c,n,h,w = x.shape

        x_lf = self.dwconv(x)
        x_hf = x - x_lf
        return x_lf,x_hf

class CascadeTransformerBlock(nn.Module):
    def __init__(self, channels,angRes,ksize = 3) -> None:
        super(CascadeTransformerBlock,self).__init__() 
        self.angRes = angRes

        self.intraT = IntraFrequencyTransformer(self.angRes, channels)
        self.interT = InterFrequencyTransformer(self.angRes, channels)
        
        # self.decompose = FrequencyDecomposeBlock(channels,channels,ksize=ksize)

    def forward(self,input):
        buffer_lf, buffer_hf = input[0],input[1]
        # deep_lf, deep_hf = self.decompose(buffer_lf+buffer_hf)

        intra_lf = self.intraT(buffer_lf) + buffer_lf
        intra_hf = self.intraT(buffer_hf) + buffer_hf

        inter_hf = self.interT(intra_hf,intra_lf) + buffer_hf
        inter_lf = self.interT(intra_lf,intra_hf) + buffer_lf

        
        # out_hf = buffer_hf + deep_hf
        return inter_lf,inter_hf
class IntraFrequencyTransformer(nn.Module):
    def __init__(self, angRes, channels):
        super(IntraFrequencyTransformer, self).__init__()
        self.angRes = angRes
        self.epi_trans = IntraDoubleTripleAttention(channels, self.angRes)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, buffer):
        shortcut = buffer
        [_, _, _, h, w] = buffer.size()
        self.epi_trans.mask_field = [self.angRes * 2, 11]

        # Horizontal
        # buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        # buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) 
        # + shortcut

        # # Vertical
        # buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        # buffer = self.epi_trans(buffer)
        # buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        # buffer = self.conv(buffer) + shortcut

        return buffer   

class IntraFreqAttentionV2(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(IntraFreqAttentionV2, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)

        epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer
class InterFrequencyTransformer(nn.Module):
    def __init__(self, angRes, channels):
        super(InterFrequencyTransformer, self).__init__()
        self.angRes = angRes
        self.epi_trans = InterDoubleTripleAttention(channels, self.angRes)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, fea_lf,fea_hf):
        shortcut = fea_lf
        [_, _, _, h, w] = fea_lf.size()
        self.epi_trans.mask_field = [self.angRes * 2, 11]

        # Horizontal
        # buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        # buffer = self.epi_trans(buffer)
        # buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        # buffer = self.conv(buffer) + shortcut

        # Vertical
        fea_lf = rearrange(fea_lf, 'b c (u v) h w -> b c (v u) w h', u=self.angRes, v=self.angRes)
        fea_hf = rearrange(fea_hf, 'b c (u v) h w -> b c (v u) w h', u=self.angRes, v=self.angRes)
        fea_lf = self.epi_trans(fea_lf,fea_hf)
        fea_lf = rearrange(fea_lf, 'b c (v u) w h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(fea_lf) 
        # + shortcut

        return buffer   
    


class InterFreqAttentionV2(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(InterFreqAttentionV2, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.linear_qk = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, x_lf,x_hf):
        [_, _, n, v, w] = x_lf.size()
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(x_lf.device)

        epi_token = rearrange(x_lf, 'b c n v w -> (v w) (b n) c')
        epi_kv = rearrange(x_hf, 'b c n v w -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)
        epi_kv = self.linear_qk(epi_kv)
        epi_q = self.norm(epi_token)
        epi_kv_norm = self.norm(epi_kv)
        epi_token = self.attention(query=epi_q,
                                   key=epi_kv_norm,
                                   value=epi_kv,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer

class InterFreqAttention(nn.Module):
    def __init__(self, channels, spa_dim,use_mask = True, num_heads=8, dropout=0.):
        super(InterFreqAttention, self).__init__()
        self.use_mask = use_mask
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.linear_kv = nn.Linear(channels, spa_dim*2, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        # self.norm_kv = nn.LayerNorm(spa_dim)
        # self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        # nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        # self.attention.out_proj.bias = None
        # self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)


    def forward(self, fea_lf,fea_hf):
        [_, _, n, v, w] = fea_lf.size()
        # attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)

        token_q = rearrange(fea_lf, 'b c n v w -> (v w) (b n) c')
        token_kv = rearrange(fea_hf, 'b c n v w -> (v w) (b n) c')
        token_q  = self.linear_in(token_q)
        token_kv = self.linear_kv(token_kv)
        
        token_k,token_v = torch.chunk(token_kv,2,-1)

        token_q = self.norm(token_q)
        token_k = self.norm(token_k)

        token_q = rearrange(token_q, '(v w) (b n) c -> b n (v w) c', n = n, v=v,w=w)
        token_k = rearrange(token_k, '(v w) (b n) c -> b n c (v w)', n = n, v=v,w=w)
        token_v = rearrange(token_v, '(v w) (b n) c -> b n (v w) c', n = n, v=v,w=w)
        
        score = token_q @ token_k
        if self.use_mask:
            size = score.shape[-1]
            att_mask = torch.tril(torch.ones(size, size, device=fea_lf.device))
            # att_mask = torch.tril(torch.ones(n, n, device=.device))
            score = score.masked_fill(att_mask == 0, -1e15)
        score = F.softmax(score,-1)
        att = score @ token_v

        att = rearrange(att ,'b n (v w) c -> (b n) (v w) c', v=v, w=w, n=n)
        token_v = rearrange(token_v ,'b n (v w) c -> (b n) (v w) c', v=v, w=w, n=n)
        epi_token = self.feed_forward(att) + token_v
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, ' (b n) (v w) c -> b c n v w', v=v, w=w, n=n)

        return buffer


class IntraFreqAttention(nn.Module):
    def __init__(self, channels, spa_dim,use_mask = True, num_heads=8, dropout=0.):
        super(IntraFreqAttention, self).__init__()
        self.use_mask = use_mask
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        # self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        # nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        # self.attention.out_proj.bias = None
        # self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)


    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        # attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)

        epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)
        epi_token_norm = self.norm(epi_token)
        epi_token_v = rearrange(epi_token, '(v w) (b n) c -> b n (v w) c', n = n, v=v,w=w)

        epi_token_q = rearrange(epi_token_norm, '(v w) (b n) c  -> b n (v w) c', n = n, v=v,w=w)
        epi_token_k = rearrange(epi_token_norm, '(v w) (b n) c  -> b n c (v w)', n = n, v=v,w=w)
        
        score = epi_token_q @ epi_token_k
        if self.use_mask:
            size = score.shape[-1]
            att_mask = torch.tril(torch.ones(size, size, device=buffer.device))
            # att_mask = torch.tril(torch.ones(n, n, device=buffer.device))
            score = score.masked_fill(att_mask == 0, -1e15)
        score = F.softmax(score,-1)
        att = score @ epi_token_v

        att = att + epi_token_v
        att = rearrange(att ,'b n (v w) c -> (b n) (v w) c', v=v, w=w, n=n)
        epi_token_v = rearrange(epi_token_v ,'b n (v w) c -> (b n) (v w) c', v=v, w=w, n=n)
        epi_token = self.feed_forward(att) + epi_token_v
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, ' (b n) (v w) c -> b c n v w', v=v, w=w, n=n)

        return buffer

class TrippleAttention(nn.Module):
    def __init__(self, channels,angRes) -> None:
        super(TrippleAttention,self).__init__()
        dims = channels
        self.angRes = angRes
        spa_dim,ang_dim,fre_dim = dims,dims,dims
        self.spa_att = SpatialAttention(channels,spa_dim)
        self.ang_att = AngularAttention(channels,ang_dim,angRes)
        
        self.fre_att = FrequencyAttention(channels,fre_dim,angRes)
        self.c_mlp = nn.Sequential(
            nn.LayerNorm(spa_dim+ang_dim+fre_dim),
            nn.Linear(spa_dim+ang_dim+fre_dim, channels, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.01),
            nn.Linear(channels, channels, bias=False),
            nn.Dropout(0.01)
        )
        
    def forward(self,x):
        b,c,n,h,w = x.shape
        x_res = x


        s_fea = self.spa_att(x_res)
        a_fea = self.ang_att(x_res)
        f_fea = self.fre_att(x_res)


        cat_fea = torch.cat([s_fea,a_fea,f_fea],1)
       
        cat_fea = rearrange(cat_fea,'b c n h w -> (b h w) n c')
        out_fea = self.c_mlp(cat_fea)
        out_fea = rearrange(out_fea,'(b h w) n c -> b c n h w',h=h,w=w) + x_res
        return out_fea
class SpatialAttention(nn.Module):
    def __init__(self, channels,dims) -> None:
        super(SpatialAttention,self).__init__()
        self.spa_q = nn.Conv2d(channels,dims,1,1,0,bias=False)
        self.spa_kv = nn.Conv2d(channels,dims*2,kernel_size=1,stride=2,padding=0,bias=False)
    def forward(self,x):
        b,c,n,h,w = x.shape
        spa_x = rearrange(x,'b c n h w -> (b n ) c h w')
        spa_kv = self.spa_kv(spa_x)
        spa_q= self.spa_q(spa_x)
        spa_k,spa_v = torch.chunk(spa_kv,2,1)
        unfold_q = rearrange(spa_q,'(b n ) c h w-> b (h w) (c n)',n=n)
        unfold_k = rearrange(spa_k,'(b n ) c h w -> b (c n) (h w)',n=n)
        unfold_v = rearrange(spa_v,'(b n ) c h w -> b (h w) (c n)',n=n)

        unfold_q = torch.nn.functional.normalize(unfold_q, dim=-1)
        unfold_k = torch.nn.functional.normalize(unfold_k, dim=-2)
        spa_att = unfold_q@unfold_k# (b h w) c (u v), (b h w) (u v) c -> (b h w) c c
        spa_att = F.softmax(spa_att,-1)
        out_att = spa_att  @ unfold_v       
        out_att = rearrange(out_att,'b (h w) (c n) -> b c n h w',n=n,h=h,w=w)
        return out_att
class FrequencyAttention(nn.Module):
    def __init__(self, channels,dims,angRes,use_mask = False) -> None:
        super(FrequencyAttention,self).__init__()
        self.angRes = angRes
        self.use_mask = use_mask
        self.ang_qkv = nn.Conv2d(channels,dims*3,1,1,0,bias=False)
    def forward(self,x):
        b,c,n,h,w = x.shape
        fre_x = rearrange(x,'b c n h w -> (b n) c h w')
        fre_x = self.ang_qkv(fre_x)
        fre_q,fre_k,fre_v = torch.chunk(fre_x,3,1)

        unfold_q = rearrange(fre_q,'(b n) c h w -> b c (n h w)',n=n)
        unfold_k = rearrange(fre_k,'(b n) c h w -> b (n h w) c',n=n)
        unfold_v = rearrange(fre_v,'(b n) c h w -> b c (n h w)',n=n)

        unfold_q = torch.nn.functional.normalize(unfold_q, dim=-1)
        unfold_k = torch.nn.functional.normalize(unfold_k, dim=-2)
        fre_att = unfold_q@unfold_k# (b h w) c (u v), (b h w) (u v) c -> (b h w) c c
        if self.use_mask:
            size = unfold_k.shape[-1]
            att_mask = torch.tril(torch.ones(size, size, device=x.device))
            fre_att = fre_att.masked_fill(att_mask == 0, -1e15)
        fre_att = F.softmax(fre_att,-1)
        out_att = fre_att  @ unfold_v       
        out_att = rearrange(out_att,'b c (n h w) -> b c n h w',n=n, h=h,w=w)
        return out_att

class InterFrequencyAttention(nn.Module):
    def __init__(self, channels,dims,angRes,use_mask = False) -> None:
        super(InterFrequencyAttention,self).__init__()
        self.angRes = angRes
        self.use_mask = use_mask
        self.to_q = nn.Conv2d(channels,dims,1,1,0,bias=False)
        self.to_kv = nn.Conv2d(channels,dims*2,1,1,0,bias=False)
    def forward(self,lf,hf):
        b,c,n,h,w = lf.shape
        fre_hf = rearrange(hf,'b c n h w -> (b n) c h w')
        fre_hf = self.to_kv(fre_hf)
        lf = rearrange(lf,'b c n h w -> (b n) c h w')
        lf = self.to_q(lf)
        
        fre_k,fre_v = torch.chunk(fre_hf,2,1)

        unfold_q = rearrange(lf,'  (b n) c h w -> b c (n h w)',n=n)
        unfold_k = rearrange(fre_k,'(b n) c h w -> b (n h w) c',n=n)
        unfold_v = rearrange(fre_v,'(b n) c h w -> b c (n h w)',n=n)

        unfold_q = torch.nn.functional.normalize(unfold_q, dim=-1)
        unfold_k = torch.nn.functional.normalize(unfold_k, dim=-2)
        fre_att = unfold_q@unfold_k# (b h w) c (u v), (b h w) (u v) c -> (b h w) c c
        if self.use_mask:
            size = fre_att.shape[-1]
            att_mask = torch.tril(torch.ones(size, size, device=lf.device))
            fre_att = fre_att.masked_fill(att_mask == 0, -1e15)
        fre_att = F.softmax(fre_att,-1)
        out_att = fre_att  @ unfold_v       
        out_att = rearrange(out_att,'b c (n h w) -> b c n h w',n=n, h=h,w=w)
        return out_att
class AngularAttention(nn.Module):
    def __init__(self, channels,dims,angRes) -> None:
        super(AngularAttention,self).__init__()
        self.angRes = angRes
        self.ang_qkv = nn.Conv2d(channels,dims*3,1,1,0,bias=False)
    def forward(self,x):
        b,c,n,h,w = x.shape
        ang_x = rearrange(x,'b c (u v) h w -> (b h w) c u v',u=self.angRes,v=self.angRes)
        ang_x = self.ang_qkv(ang_x)
        ang_q,ang_k,ang_v = torch.chunk(ang_x,3,1)

        unfold_q = rearrange(ang_q,'(b h w) c u v -> b (u v) (c h w)', h= h,w=w)
        unfold_k = rearrange(ang_k,'(b h w) c u v -> b (c h w) (u v)', h= h,w=w)
        unfold_v = rearrange(ang_v,'(b h w) c u v -> b (u v) (c h w)', h= h,w=w)

        unfold_q = torch.nn.functional.normalize(unfold_q, dim=-1)
        unfold_k = torch.nn.functional.normalize(unfold_k, dim=-2)
        ang_att = unfold_q@unfold_k# (b h w) c (u v), (b h w) (u v) c -> (b h w) c c
        ang_att = F.softmax(ang_att,-1)
        out_att = ang_att  @ unfold_v       
        out_att = rearrange(out_att,'b (u v) (c h w) -> b c (u v) h w',u=self.angRes,v=self.angRes,h=h,w=w)
        return out_att
    

class SpatialAngularAttention(nn.Module):
    def __init__(self, channels,dims,angRes,use_mask ) -> None:
        super(SpatialAngularAttention,self).__init__()
        self.angRes = angRes
        self.use_mask = use_mask
        # self.to_qkv = nn.Conv2d(channels,dims*3,1,1,0,bias=False)
    def forward(self,token_q,token_k,token_v):

        b,c,n,h,w = token_q.shape
        # print('token size is {}'.format(token_q.shape))
        buffer = token_v

        # spaang_x = rearrange(x,'b c (u v) h w -> (b u v) c h w',u=self.angRes,v=self.angRes)
        # spaang_x = self.to_qkv(spaang_x)
        # spaang_q,spaang_k,spaang_v = torch.chunk(spaang_x,3,1)

        unfold_q = rearrange(token_q,'b c (u v) h w -> b (h u) (c v w)',u=self.angRes,v=self.angRes)
        unfold_k = rearrange(token_k,'b c (u v) h w -> b (c v w) (h u)',u=self.angRes,v=self.angRes)
        unfold_v = rearrange(token_v,'b c (u v) h w -> b (h u) (c v w)',u=self.angRes,v=self.angRes)

        unfold_q = torch.nn.functional.normalize(unfold_q, dim=-1)
        unfold_k = torch.nn.functional.normalize(unfold_k, dim=-2)
        spaang_att = unfold_q@unfold_k# (b h w) c (u v), (b h w) (u v) c -> (b h w) c c
        if self.use_mask:
            size = unfold_k.shape[-1]
            att_mask = torch.tril(torch.ones(size, size, device=token_v.device))
            spaang_att = spaang_att.masked_fill(att_mask == 0, -1e15)
        spaang_att = F.softmax(spaang_att,-1)
        out_att = spaang_att  @ unfold_v       
        out_att = rearrange(out_att,'b (h u) (c v w) -> b c (u v) h w',u=self.angRes,v=self.angRes,h=h,w=w) + buffer
        return out_att

class AngularFrequencyAttention(nn.Module):
    def __init__(self, channels,dims,angRes,use_mask ) -> None:
        super(AngularFrequencyAttention,self).__init__()
        self.angRes = angRes
        self.use_mask = use_mask
        # self.to_qkv = nn.Conv2d(channels,dims*3,1,1,0,bias=False)

        
    def forward(self,token_q,token_k,token_v):

        b,c,n,h,w = token_q.shape
        buffer = token_v

        # angfre_x = rearrange(x,'b c (u v) h w -> (b u v) c h w',u=self.angRes,v=self.angRes)
        # angfre_x = self.to_qkv(angfre_x)
        # angfre_q,angfre_k,angfre_v = torch.chunk(angfre_x,3,1)

       

        unfold_q = rearrange(token_q,'b c (u v) h w -> b (c u) (h v w)',u=self.angRes,v=self.angRes)
        unfold_k = rearrange(token_k,'b c (u v) h w -> b (h v w) (c u)',u=self.angRes,v=self.angRes)
        unfold_v = rearrange(token_v,'b c (u v) h w -> b (c u) (h v w)',u=self.angRes,v=self.angRes)
        
        unfold_q = torch.nn.functional.normalize(unfold_q, dim=-1)
        unfold_k = torch.nn.functional.normalize(unfold_k, dim=-2)
        spaang_att = unfold_q@unfold_k# (b h w) c (u v), (b h w) (u v) c -> (b h w) c c
        if self.use_mask:
            size = unfold_k.shape[-1]
            att_mask = torch.tril(torch.ones(size, size, device=token_q.device))
            spaang_att = spaang_att.masked_fill(att_mask == 0, -1e15)
        spaang_att = F.softmax(spaang_att,-1)
        out_att = spaang_att  @ unfold_v       
        out_att = rearrange(out_att,'b (c u) (h v w) -> b c (u v) h w',u=self.angRes,v=self.angRes,h=h,w=w) + buffer
        return out_att
    
class SpatialFrequencyAttention(nn.Module):
    def __init__(self, channels,dims,angRes,use_mask ) -> None:
        super(SpatialFrequencyAttention,self).__init__()
        self.angRes = angRes
        self.use_mask = use_mask
        # self.to_qkv = nn.Conv2d(channels,dims*3,1,1,0,bias=False)
    def forward(self,token_q,token_k,token_v):

        b,c,n,h,w = token_q.shape
        buffer = token_v

        # spafre_x = self.to_qkv(spafre_x)
        # spafre_q,spafre_k,spafre_v = torch.chunk(spafre_x,3,1)

        unfold_q = rearrange(token_q,'b c (u v) h w -> b (c h ) (w u v)',u=self.angRes,v=self.angRes)
        unfold_k = rearrange(token_k,'b c (u v) h w -> b (w u v) (c h )',u=self.angRes,v=self.angRes)
        unfold_v = rearrange(token_v,'b c (u v) h w -> b (c h ) (w u v)',u=self.angRes,v=self.angRes)

        unfold_q = torch.nn.functional.normalize(unfold_q, dim=-1)
        unfold_k = torch.nn.functional.normalize(unfold_k, dim=-2)
        spafre_att = unfold_q@unfold_k# (b h w) c (u v), (b h w) (u v) c -> (b h w) c c
        if self.use_mask:
            size = unfold_k.shape[-1]
            att_mask = torch.tril(torch.ones(size, size, device=token_v.device))
            spafre_att = spafre_att.masked_fill(att_mask == 0, -1e15)
        spafre_att = F.softmax(spafre_att,-1)
        out_att = spafre_att  @ unfold_v 
        out_att = rearrange(out_att,'b (c h ) (w u v) -> b c (u v) h w',u=self.angRes,v=self.angRes,h=h,w=w) + buffer
        return out_att
    
class DoubleTripleAttention(nn.Module):
    def __init__(self, channels,angRes,use_mask = False) -> None:
        super(DoubleTripleAttention,self).__init__()
        dims = channels*2
        self.angRes = angRes
        spa_dim,ang_dim,fre_dim = dims//2,dims//2,dims
        self.spafre_att = SpatialFrequencyAttention(channels,ang_dim,angRes,use_mask)
        self.angfre_att = AngularFrequencyAttention(channels,spa_dim,angRes,use_mask)
        
        self.spaang_att = SpatialAngularAttention(channels,fre_dim,angRes,use_mask)
        self.c_mlp = nn.Sequential(
            nn.LayerNorm(spa_dim+ang_dim+fre_dim),
            nn.Linear(spa_dim+ang_dim+fre_dim, channels, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.01),
            nn.Linear(channels, channels, bias=False),
            nn.Dropout(0.01)
        )
        
    def forward(self,x):
        b,c,n,h,w = x.shape
        x_res = x


        sf_fea = self.spafre_att(token_q = x_res, token_k = x_res, token_v = x_res)
        af_fea = self.angfre_att(token_q = x_res, token_k = x_res, token_v = x_res)
        sa_fea = self.spaang_att(token_q = x_res, token_k = x_res, token_v = x_res)


        cat_fea = torch.cat([sf_fea,af_fea,sa_fea],1)
       
        cat_fea = rearrange(cat_fea,'b c n h w -> (b h w) n c')
        out_fea = self.c_mlp(cat_fea) + cat_fea
        out_fea = rearrange(out_fea,'(b h w) n c -> b c n h w',h=h,w=w) + x_res
        return out_fea
    
class IntraDoubleTripleAttention(nn.Module):
    def __init__(self, channels,angRes,use_mask = True) -> None:
        super(IntraDoubleTripleAttention,self).__init__()
        dims = channels
        self.angRes = angRes
        spa_dim,ang_dim,fre_dim = dims//2,dims//2,dims
        self.linearIn = nn.Linear(channels,spa_dim+ang_dim+fre_dim,bias=False)
        self.spafre_att = SpatialFrequencyAttention(channels,ang_dim,angRes,use_mask)
        self.angfre_att = AngularFrequencyAttention(channels,spa_dim,angRes,use_mask)
        self.spaang_att = SpatialAngularAttention(channels,fre_dim,angRes,use_mask)
        self.c_mlp = nn.Sequential(
            nn.LayerNorm(spa_dim+ang_dim+fre_dim),
            nn.Linear(spa_dim+ang_dim+fre_dim, spa_dim+ang_dim+fre_dim, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.01),
            nn.Linear(spa_dim+ang_dim+fre_dim, spa_dim+ang_dim+fre_dim, bias=False),
            nn.Dropout(0.01)
        )

        self.out = nn.Linear(spa_dim+ang_dim+fre_dim,channels,bias=False)
        
    def forward(self,x):
        b,c,n,h,w = x.shape
        x_res = rearrange(x, 'b c (u v) h w -> (b u v) (h w)  c', u = self.angRes,v=self.angRes)
        x_token = self.linearIn(x_res)
        x_token = rearrange(x_token,'(b u v) (h w)  c -> b c (u v) h w',h=h,w=w,u=self.angRes,v=self.angRes)
        sf_token,af_token,sa_token1,sa_token2 = torch.chunk(x_token,4,1)
        sa_token = torch.cat([sa_token1,sa_token2],1)


        sf_fea = self.spafre_att(token_q = sf_token, token_k = sf_token, token_v = sf_token)
        af_fea = self.angfre_att(token_q = af_token, token_k = af_token, token_v = af_token)
        sa_fea = self.spaang_att(token_q = sa_token, token_k = sa_token, token_v = sa_token)

        cat_fea = torch.cat([sf_fea,af_fea,sa_fea],1)
        cat_fea = rearrange(cat_fea,'b c (u v) h w -> (b u v) (h w) c',u=self.angRes,v=self.angRes)
       
        out_fea = self.c_mlp(cat_fea) + cat_fea
        out_fea = self.out(out_fea)
        out_fea = rearrange(out_fea,'(b u v) (h w)  c -> b c (u v) h w',h=h,w=w,u=self.angRes,v=self.angRes)
        return out_fea

class InterDoubleTripleAttention(nn.Module):
    def __init__(self, channels,angRes,use_mask = True) -> None:
        super(InterDoubleTripleAttention,self).__init__()
        dims = channels
        self.angRes = angRes
        spa_dim,ang_dim,fre_dim = dims//2,dims//2,dims
        self.linearIn = nn.Linear(channels,spa_dim+ang_dim+fre_dim,bias=False)
        self.linearkv = nn.Linear(channels,spa_dim+ang_dim+fre_dim,bias=False)
        self.spafre_att = SpatialFrequencyAttention(channels,ang_dim,angRes,use_mask)
        self.angfre_att = AngularFrequencyAttention(channels,spa_dim,angRes,use_mask)
        self.spaang_att = SpatialAngularAttention(channels,fre_dim,angRes,use_mask)
        self.c_mlp = nn.Sequential(
            nn.LayerNorm(spa_dim+ang_dim+fre_dim),
            nn.Linear(spa_dim+ang_dim+fre_dim, spa_dim+ang_dim+fre_dim, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.01),
            nn.Linear(spa_dim+ang_dim+fre_dim, spa_dim+ang_dim+fre_dim, bias=False),
            nn.Dropout(0.01)
        )

        self.out = nn.Linear(spa_dim+ang_dim+fre_dim,channels,bias=False)
        
    def forward(self,x_lf,x_hf):
        b,c,n,h,w = x_hf.shape
        x_hf = rearrange(x_hf, 'b c (u v) h w -> (b u v) (h w)  c', u = self.angRes,v=self.angRes)
        x_lf = rearrange(x_lf, 'b c (u v) h w -> (b u v) (h w)  c', u = self.angRes,v=self.angRes)
        q_token = self.linearIn(x_hf)
        kv_token = self.linearkv(x_lf)
        q_token = rearrange(q_token,'(b u v) (h w)  c -> b c (u v) h w',h=h,w=w,u=self.angRes,v=self.angRes)
        kv_token = rearrange(kv_token,'(b u v) (h w)  c -> b c (u v) h w',h=h,w=w,u=self.angRes,v=self.angRes)
        sf_qtoken,af_qtoken,sa_qtoken1,sa_qtoken2 = torch.chunk(q_token,4,1)
        sf_kvtoken,af_kvtoken,sa_kvtoken1,sa_kvtoken2 = torch.chunk(kv_token,4,1)
        sa_qtoken = torch.cat([sa_qtoken1,sa_qtoken2],1)
        sa_kvtoken = torch.cat([sa_kvtoken1,sa_kvtoken2],1)


        sf_fea = self.spafre_att(token_q = sf_qtoken, token_k = sf_kvtoken, token_v = sf_kvtoken)
        af_fea = self.angfre_att(token_q = af_qtoken, token_k = af_kvtoken, token_v = af_kvtoken)
        sa_fea = self.spaang_att(token_q = sa_qtoken, token_k = sa_kvtoken, token_v = sa_kvtoken)

        cat_fea = torch.cat([sf_fea,af_fea,sa_fea],1)
        cat_fea = rearrange(cat_fea,'b c (u v) h w -> (b u v) (h w) c',u=self.angRes,v=self.angRes)
       
        out_fea = self.c_mlp(cat_fea) + cat_fea
        out_fea = self.out(out_fea)
        out_fea = rearrange(out_fea,'(b u v) (h w)  c -> b c (u v) h w',h=h,w=w,u=self.angRes,v=self.angRes)
        return out_fea
    
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
def FormOutput(intra_fea):
    b, n, c, h, w = intra_fea.shape
    angRes = int(sqrt(n+1))
    intra_fea = intra_fea.squeeze(2).contiguous().permute(0,2,3,1).view(b,h,w,angRes,angRes)
    
    out= LFreshape(intra_fea, angRes)

    return out

import torchvision.models as models

class StyleLoss(nn.Module):
    def __init__(self, layernum = 1):
        super(StyleLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.vgg.to('cuda').eval()
        self.conv = nn.Sequential(*list(self.vgg.children())[:layernum])
        self.criterion = torch.nn.L1Loss()
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, SR,HR):
        sr_lf,hr_lf = self.conv(SR.repeat(1,3,1,1)),self.conv(HR.repeat(1,3,1,1))
        sr_hf,hr_hr = SR - sr_lf , HR - hr_lf
        
        sr_lf_gm, hr_lf_gm = gram_matrix(sr_lf),gram_matrix(hr_lf)
        sr_hf_gm, hr_hf_gm = gram_matrix(sr_hf),gram_matrix(hr_hr)
        lf_loss = self.criterion(sr_lf_gm,hr_lf_gm)
        hf_loss = self.criterion(sr_hf_gm,hr_hf_gm)

        return lf_loss + hf_loss

# 定义 gram 矩阵计算函数
def gram_matrix(input):
    batch_size, channels, height, width = input.size()
    features = input.view(batch_size * channels, height * width)
    gram_matrix = torch.mm(features, features.t())
    gram_matrix /= (batch_size * channels * height * width)
    return gram_matrix


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()
        self.frequency_loss = StyleLoss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)
        # fre_loss = self.frequency_loss(SR,HR)
        #  + fre_loss*10
        return loss


def weights_init(m):
    pass