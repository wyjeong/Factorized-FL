import torch
import random
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image, ImageOps
from misc.utils import *

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.n_workers = 1
        self.client_id = None

    def switch(self, client_id):
        if not self.client_id == client_id:
            self.client_id = client_id
            self.partition = Dataset(self.args, mode='partition', client_id=client_id)
            self.pa_loader = torch.utils.data.DataLoader (dataset=self.partition, batch_size=self.args.batch_size, 
                shuffle=True, num_workers=self.n_workers, drop_last=True, pin_memory=True)     
            self.test = Dataset(self.args, mode='test', client_id=client_id)
            self.te_loader = torch.utils.data.DataLoader (dataset=self.test, batch_size=self.args.batch_size, 
                shuffle=False, num_workers=self.n_workers, drop_last=True, pin_memory=True)
            self.valid = Dataset(self.args, mode='val', client_id=client_id)    
            self.va_loader = torch.utils.data.DataLoader (dataset=self.valid, batch_size=self.args.batch_size, 
                shuffle=False, num_workers=self.n_workers, drop_last=True, pin_memory=True) 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, mode, client_id=-1):
        self.args = args
        self.mode = mode
        self.image_size = 32
        self.mean, self.std = self.get_stats(self.args.dataset)
        self.client_id = client_id

        if self.args.multi:
            if 'CIFAR_100' in self.args.dataset:
                self.data = torch_load(self.args.data_path, 
                                f'{self.args.dataset}_dom_{client_id}_{mode}.pt')
            else:
                dom_id = client_id//4 
                if self.mode in ['test', 'val']:
                    self.data = torch_load(self.args.data_path, 
                                    f'{self.args.dataset}_{dom_id}_{mode}.pt')
                else:
                    self.data = torch_load(self.args.data_path, 
                                    f'{self.args.dataset}_{dom_id}_{client_id}.pt')
        else:
            if self.mode in ['test', 'val']:
                self.data = torch_load(self.args.data_path, 
                                f'{self.args.dataset}_{mode}.pt')
            else:
                self.data = torch_load(self.args.data_path, 
                    f'{self.args.dataset}_{self.args.dist}_{mode}_{client_id}.pt')
        
        self.x = self.data['x']
        self.y = self.data['y']
        if self.args.permuted:
            idx = np.arange(self.args.n_clss)
            random.seed(self.args.seed+client_id)
            random.shuffle(idx)
            self.y_permutation = {clss_id: permuted_id for clss_id, permuted_id in enumerate(idx)}
        self.set_transform()
        
    def set_transform(self):
        if self.mode in ['test', 'val']:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(self.mean, self.std)])
        else:
            self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            self.transform = transforms.Compose([
                        # transforms.Grayscale(3) for MNIST
                        transforms.RandomResizedCrop(self.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([self.color_jitter], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std)])

    def get_stats(self, dataset):
        if dataset == 'CIFAR_100' or dataset == 'MULTI':
            mean = (0.507, 0.487, 0.441)
            std = (0.268, 0.257, 0.276)
        elif dataset == 'CIFAR_10':
            mean = (0.491, 0.482, 0.447)
            std = (0.247, 0.243, 0.262)
        elif dataset == 'SVHN':
            mean = (0.4380, 0.4440, 0.4730)
            std = (0.1751, 0.1771, 0.1744)
        return mean, std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.args.permuted:
            return self.transform(self.x[index]), self.y_permutation[self.y[index]] 
        else:
            return self.transform(self.x[index]), self.y[index]

