def set_template(args):
    # Set the templates here
    

        
    if args.template == 'FSRnet':
        args.model = 'FSRnet'
        args.n_feats = 64
        args.decay_every = 20
        args.epochs = 60
        args.lr = 2e-4
        args.batch_size = 8
        args.optimizer = 'ADAM'
    elif args.template == 'IINet':
        args.model = 'IINet'
        args.n_feats = 32
        args.decay_every = 10
        args.epochs = 50
        args.lr = 2e-4
        args.batch_size = 10
        args.optimizer = 'ADAM'

    

    
        
        
        
    else:
        raise ValueError('template {} is not included '.format(args.template))
        
        

   
    
        

