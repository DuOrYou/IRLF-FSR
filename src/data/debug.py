import torch
import torch.nn as nn

a = torch.ones([2,4*9,3,3])
count = 0
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        count +=1
        a[i,j,:,:] = torch.ones([3,3])*count
        
print(a[0,0,:,:])
data=a
        
splited_d = torch.chunk(data,9,axis=1) #groups
angRes=3
angs_ = []
for i in range(angRes):
    angs_.append(torch.cat(splited_d[i*angRes:(i+1)*angRes],2))
lf_rs = torch.cat(angs_[:],-1)
print(lf_rs.shape)
print(lf_rs[0,0,:,:])
# aa_h = torch.chunk(a, 2, 3)
# aa_cat_h = torch.cat(aa_h[:],1)
# aa_v = torch.chunk(aa_cat_h,2,-1)
# aa_cat = torch.cat(aa_v[:],2)
# aa_cat = torch.squeeze(aa_cat)
# print(aa_cat.shape)
# print(aa_cat[0,:,:])
# for i in range(a.shape[3]):
#     current_ = torch.cat(a[:,:,:,i,:],1)
#     aa_h = torch.cat(a[:,:,:,i,:],1)
# print(aa_h.shape)
# for h in range(a.shape[-1]):
#     aa_v = torch.cat(aa_h[:,:,:,h],2)
# print(aa_v.shape)
# aa_v = torch.unsqueeze(aa_v,-1)
# print(aa_v[0,:,:])
# print(spl_a[0][0,0,:,:])
# angs_ = []
# for i in range(2):
#     print(i)
#     angs_.append(torch.cat(spl_a[i*2:(i+1)*2],2))
# print(angs_[0][0,0,:,:])
# # print(angs_[1].shape)
# aa = torch.cat(angs_[:],-1)
# print(aa[0,0,:,:])
print('done')