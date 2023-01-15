import os
from parser import Parser
from datetime import datetime

from misc.utils import *
from modules.multiprocs import ParentProcess

def main(args):

    args = set_config(args)

    if args.model == 'factorized_fl':      
        from models.factorized_fl.server import Server
        from models.factorized_fl.client import Client
    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)

    pp = ParentProcess(args, Server, Client)
    pp.start()

def set_config(args):

    args.base_lr = 1e-3
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    args.weight_decay = 1e-6
    args.warmup_epochs = 10
    args.base_momentum = 0.99
    args.final_momentum = 1.0
    
    args.head = False if args.permuted else True

    if args.task in ['CIFAR_10_IID', 'CIFAR_100_IID', 'SVHN_IID']:
        args.multi = False
        args.n_clients = 20 
        args.frac = 1.0 
        args.batch_size = 256
        args.dist = 'iid'
        args.dataset = args.task.replace('_IID', '')
        args.n_clss = 100 if '100' in args.task else 10

    elif args.task in ['CIFAR_10_NON_IID', 'CIFAR_100_NON_IID', 'SVHN_NON_IID']:
        args.multi = False
        args.n_clients = 20 
        args.frac = 1.0 
        args.batch_size = 256
        args.dist = 'non_iid'
        args.dataset = args.task.replace('_NON_IID', '')
        args.n_clss = 100 if '100' in args.task else 10

    elif args.task in ['CIFAR_100_DOM']:
        args.multi = True
        args.head = False
        args.n_clients = 20 
        args.frac = 1.0 
        args.batch_size = 256
        args.n_clss = 5
        args.dist = 'dom'
        args.dataset = args.task.replace('_DOM', '')

    elif args.task in ['MULTI_DOM']:
        args.multi = True
        args.head = False
        args.n_clients = 20 
        args.frac = 1.0 
        args.batch_size = 256
        args.n_clss = 10
        args.dist = 'dom'
        args.dataset = args.task.replace('_DOM', '')

    if 'factorized_fl' in args.model:
        
        args.aggr = 'uvm' if 'plus' in args.model else 'u'

        if args.multi:
            args.rank = 1
            args.lambda_l1 = 5e-4
            args.tau = 0.95
            args.alpha = 20
            args.minimum_aggr = 4
        else:
            args.rank = 1
            if args.dist == 'iid':
                if args.permuted:
                    args.lambda_l1 = 5e-4
                    args.tau = 0
                    args.alpha = 1
                    args.minimum_aggr = 10
                else:
                    args.lambda_l1 = 5e-4
                    args.tau = 0
                    args.alpha = 1
                    args.minimum_aggr = 20
            else:
                if args.permuted:
                    args.lambda_l1 = 5e-4
                    args.tau = 0
                    args.alpha = 1
                    args.minimum_aggr = 10
                else:
                    args.lambda_l1 = 5e-4
                    args.tau = 0
                    args.alpha = 1
                    args.minimum_aggr = 20


    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    now_ymd = now.split('_')[0]
    now_hms = now.split('_')[1]
    
    trial = f'{args.project}/{now}_{args.task}_{args.model}' \
            if args.trial == None else f'{args.project}/{now}_{args.task}_{args.model}_{args.trial}'

    ####################################################
    args.data_path = f'{args.base_path}/data' 
    ####################################################
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    args.log_path = f'{args.base_path}/logs/{trial}'

    return args

if __name__ == '__main__':
    main(Parser().parse())










