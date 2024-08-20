# -*- coding: utf-8 -*-
import torch.cuda
import torch.distributed
from option import args
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpunum)
import torch
import torch.nn as nn
import utility
import data
import model
import loss

       
from trainer import Trainer as T
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model

    if checkpoint.ok:
        loader = data.Data(args)
        print('done load data')
        # gpu_tracker.track()
        _model = model.Model(args, checkpoint)
        # gpu_tracker.track()
        # devices = []
        # sd = _model.state_dict()
        # for v in sd.values():
        #     # if v.device not in devices:
        #     devices.append(v.device)

        # for d in devices:
        #     print(d)

        # if torch.cuda.device_count() > 1:
        #     print("Use", torch.cuda.device_count(), 'gpus')
        #     _model = nn.DataParallel(_model)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None

        t = T(args, loader, _model, _loss, checkpoint)

        # aa = t.terminate()
        while t.terminate():
            eps = t.terminate()

            # t.test()
            t.train()
            # t.val()
            # if t.ternimate()./10==0:

            # if eps%10==0 or eps ==1:

            #     t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
