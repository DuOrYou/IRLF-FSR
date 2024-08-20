from importlib import import_module
#from dataloader import MSDataLoader
import torch.utils
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import torch
import torch.utils.data
# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)  #if d has [set_scale] property then set a scle 

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only: # set train dataset 
            # datasets = []
        # for d in args.data_train:
            module_train = 'train_process' 
            m_train = import_module('data.train_data' )
            datasets = getattr(m_train, module_train)(args)
            
            
            self.loader_train = dataloader.DataLoader(
                datasets,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                prefetch_factor  = 2,
                num_workers=args.n_threads,
            )
            print('===================done traindata prepare===============')
            # self.loader_train = dataloader.DataLoader(
            #     MyConcatDataset(datasets),
            #     batch_size=args.batch_size,
            #     shuffle=True,
            #     pin_memory=not args.cpu,
            #     num_workers=args.n_threads,
            # )
            
            # module_val = 'val_process' 
            # m_val = import_module('data.val_data' )
            # datasets_val = getattr(m_val, module_val)(args)
            
            # self.loader_val = dataloader.DataLoader(
            #     datasets_val,
            #     batch_size=1,
            #     shuffle=True,
            #     pin_memory=True,
            #     prefetch_factor  = 2,
            #     num_workers=args.n_threads,
            # )
            # print('===========done valdata prepare==========')
        
            module_test = 'test_process' 
            m_test = import_module('data.test_data' )
            datasets_test = getattr(m_test, module_test)(args)
            
            self.loader_test = dataloader.DataLoader(
                datasets_test,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                
                num_workers=args.n_threads,
            )
            print('============done testdata prepare===========')
        else:
            if args.bytedepth==16:
                module_test = 'test_process' 
                m_test = import_module('data.test_img' )
                datasets_test = getattr(m_test, module_test)(args)
                
                self.loader_test = dataloader.DataLoader(
                    datasets_test,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=True,
                    
                    num_workers=args.n_threads,
                )
                print('============done byte16 prepare===========')
            else:
                # print(args.test_unknown)
                if args.without_gt:
                    module_test = 'test_process' 
                    m_test = import_module('data.test_img' )
                    datasets_test = getattr(m_test, module_test)(args)
                    # if args.n_GPUs>1:
                    #     train_sampler = torch.utils.data.distributed.DistributedSampler(datasets_test)
                    self.loader_test = dataloader.DataLoader(
                        datasets_test,
                        batch_size=1,
                        shuffle=False,
                        pin_memory=True,
                        # sampler=train_sampler,
                        
                        num_workers=args.n_threads,
                    )
                    print('============done test_unknown prepare===========')
                else:
                    
                    module_test = 'test_process' 
                    m_test = import_module('data.test_data' )
                    datasets_test = getattr(m_test, module_test)(args)
                    # if args.n_GPUs>1:
                    #     train_sampler = torch.utils.data.distributed.DistributedSampler(datasets_test)
                    self.loader_test = dataloader.DataLoader(
                        datasets_test,
                        batch_size=1,
                        shuffle=False,
                        pin_memory=True,
                        # sampler=train_sampler,
                        num_workers=args.n_threads,
                    )
                    print('============done testimgs prepare===========')
            # module_test = 'test_image' 
            # m_test = import_module('data.test_img' )
            # datasets_test_only = getattr(m_test, module_test)(args)
            
            # self.loader_test_only = dataloader.DataLoader(
            #     datasets_test_only,
            #     batch_size=1,
            #     shuffle=False,
            #     pin_memory=not args.cpu,
            #     num_workers=args.n_threads,
            # )
            # print('============done testonly prepare===========')
