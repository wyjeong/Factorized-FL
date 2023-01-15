import argparse
from misc.utils import *

class Parser:
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
       
    def set_arguments(self):
        
        self.parser.add_argument('--gpu', type=str, default='0')
        self.parser.add_argument('--seed', type=int, default=1234)

        self.parser.add_argument('--model', type=str, default=None)
        self.parser.add_argument('--task', type=str, default=None)
        
        self.parser.add_argument('--backbone', type=str, default='resnet9')
        self.parser.add_argument('--n-workers', type=int, default=None)
        self.parser.add_argument('--n-clients', type=int, default=None)
        self.parser.add_argument('--n-rnds', type=int, default=None)
        self.parser.add_argument('--n-eps', type=int, default=None)
        self.parser.add_argument('--c-rnds', type=int, default=None)
        self.parser.add_argument('--frac', type=float, default=None)
        self.parser.add_argument('--lr', type=float, default=None)

        self.parser.add_argument('--permuted', type=str2bool, default=False)
        self.parser.add_argument('--head', type=str2bool, default=False)
        # self.parser.add_argument('--aggr', type=str2bool, default=True)
        self.parser.add_argument('--trial', type=str, default=None)
        self.parser.add_argument('--project', type=str, default='')
        self.parser.add_argument('--base-path', type=str, default='')


        self.parser.add_argument('--aggr', type=str, default='u')


    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
