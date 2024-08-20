from option import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpunum)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
import numpy as np
import utility
import data
import model
import loss

from trainer import Trainer
# from tester_unknown import Tester
from tester import Tester 
import random
# torch.manual_seed(args.seed)

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



# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)  
# np.random.seed(args.seed)  # Numpy module.
# random.seed(args.seed)  # Python random module.
# torch.manual_seed(args.seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

checkpoint = utility.checkpoint(args)
if args.nonuniform_N :
    uniform = 'UN'
    if args.is_sisr:
        args.pre_train = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/AISR_MDSR/src/experiment',
                                args.model,'{}_{}x{}b{}ep{}lr{}uv{}'.format(uniform,args.model,args.scale,args.batch_size,args.epochs,args.lr,args.train_uv),
                                'model/model_latest.pt')
    else:
        args.pre_train = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/AISR_MDSR/src/experiment', 
                                args.model,'{}_{}x{}b{}ep{}lr{}'.format(uniform,args.model,args.scale,args.batch_size,args.epochs,args.lr),
                                'model/model_latest.pt')
else:

    if args.is_sisr:
        args.pre_train = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/AISR_MDSR/src/experiment',
                                args.model,'{}x{}b{}ep{}lr{}uv{}'.format(args.model,args.scale,args.batch_size,args.epochs,args.lr,args.train_uv),
                                'model/model_latest.pt')
    else:
        args.pre_train = os.path.join('/mnt/sda/duyou/Project_LFSR/our_method/AISR_MDSR/src/experiment', 
                                args.model,'{}x{}b{}ep{}lr{}'.format(args.model,args.scale,args.batch_size,args.epochs,args.lr),
                                'model/model_latest.pt')
print(args.pre_train)
# args.pre_train = '/mnt/sda/duyou/Project_LFSR/our_method/AISR_MDSR/src/experiment/{}/model/model_latest.pt'.format(args.model)
args.test_only = True
# args.cpu = True
args.save_results= True
# args.cpu = True
args.test_unknown= True
args.without_gt = True
# args.defaultname = 'img029'
def main():
    global model
    
        
    loader = data.Data(args)
    print('done load test data')
    _model = model.Model(args, checkpoint)
    # print(_model)
    t = Tester(args, loader, _model, checkpoint)
    
    t.test()

    checkpoint.done()

if __name__ == '__main__':
    main()
