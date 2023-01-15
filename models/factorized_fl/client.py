import time
import torch
import numpy as np

from misc.utils import *
from modules.nets import *
from modules.federated import ClientModule

class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.encoder = ResNet9(factorize=True, l1=self.args.lambda_l1, rank=self.args.rank).cuda(g_id) 
        self.head = Head(self.args.n_clss, factorize=True, l1=self.args.lambda_l1, rank=self.args.rank).cuda(g_id) 
        self.parameters = list(self.encoder.parameters()) + list(self.head.parameters())

    def init_state(self):
        self.optimizer = torch.optim.SGD(self.parameters,
                            lr=self.args.base_lr,
                            momentum=self.args.momentum_opt,
                            weight_decay=self.args.weight_decay)
        
        self.log = { 'lr': [],'train_lss': [],
                        'ep_lss': [],'ep_acc_1': [],'ep_acc_5': [],
                            'ep_u': [],'ep_v': [],'ep_m': [],
                                'rnd_lss': [],'rnd_acc_1': [],'rnd_acc_5': [],
                                    'rnd_u': [],'rnd_v': [],'rnd_m': [], 'n_params':[], 'comm_cost':[]}

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'encoder': get_state_dict(self.encoder),
            'head': get_state_dict(self.head),
            'ep': self.curr_rnd,
            'log': self.log,
        })
        if (self.curr_rnd+1)%25 == 0 or self.curr_rnd==0:
            self.logger.print(f'model saved')
            torch_save(self.args.checkpt_path, f'weight_{self.client_id}_{self.curr_rnd+1}.pt', {
                'encoder': get_state_dict(self.encoder),
                'head': get_state_dict(self.head),
            })    

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.encoder, loaded['encoder'], self.gpu_id)
        set_state_dict(self.head, loaded['head'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']
    
    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        if self.curr_rnd == 0:
            self.noise = self.sd['noise'].clone().cuda(self.gpu_id)
            set_state_dict(self.encoder, self.sd['init']['encoder'], self.gpu_id, skip_bn=True)
            set_state_dict(self.head, self.sd['init']['head'], self.gpu_id, skip_bn=True)
        else:
            set_state_dict(self.encoder, self.sd[f'{self.client_id}_encoder'], self.gpu_id, skip_bn=True)
            if self.args.head:
                set_state_dict(self.head, self.sd['init']['head'], self.gpu_id, skip_bn=True)

    def on_round_begin(self, curr_rnd):
        self.train()
        self.transfer_to_server()

    def train(self):
        for ep in range(self.args.n_eps):
            st = time.time()
            self.encoder.train()
            self.head.train()
            for i, batch in enumerate(self.loader.pa_loader):
                x_batch, y_batch = batch
                x_batch = x_batch.cuda(self.gpu_id)
                y_batch = y_batch.cuda(self.gpu_id)
                y_hat = self.head(self.encoder(x_batch))
                lss = F.cross_entropy(y_hat, y_batch)
                for name, param in self.encoder.state_dict().items():
                    if 'mask' in name:
                        lss += torch.norm(param.float(), 1) * self.args.lambda_l1
                for name, param in self.head.state_dict().items():
                    if 'mask' in name:
                        lss += torch.norm(param.float(), 1) * self.args.lambda_l1
                self.optimizer.zero_grad()
                lss.backward()
                self.optimizer.step()
            acc_1, acc_5, lss = self.evaluate()
            u,v,m = self.get_n_params(self.encoder)
            if self.args.head:
                _u,_v,_m = self.get_n_params(self.head)
                u += _u 
                v += _v 
                m += _m 
            n_params = u+v+m
            if self.args.aggr == 'u':
                comm_cost = u
            elif self.args.aggr == 'uv':
                comm_cost = u + v
            elif self.args.aggr == 'um':
                comm_cost = u + m
            elif self.args.aggr == 'uvm':
                comm_cost = u + v + m
            self.logger.print(
                f'rnd:{self.curr_rnd+1}, ep:{ep+1}, loss:{lss.item():.3f}, acc_1:{acc_1:.2f}, acc_5:{acc_5:.2f}, '
                                            +f'n_params:{n_params} comm_cost:{comm_cost} ({time.time()-st:.1f}s)')
            self.log['train_lss'].append(lss.item())
            self.log['ep_acc_1'].append(acc_1)
            self.log['ep_acc_5'].append(acc_5)
            self.log['ep_u'].append(u)
            self.log['ep_v'].append(v)
            self.log['ep_m'].append(m)

        self.log['comm_cost'].append(comm_cost*(self.curr_rnd+1))
        self.log['rnd_acc_1'].append(acc_1)
        self.log['rnd_acc_5'].append(acc_5)
        self.log['rnd_u'].append(u)
        self.log['rnd_v'].append(v)
        self.log['rnd_m'].append(m)
        self.log['n_params'].append(n_params)
        self.save_log()

    @torch.no_grad()
    def transfer_to_server(self):
        z = self.encoder(self.noise)
        self.sd[self.client_id] = {
            'encoder': get_state_dict(self.encoder),
            'train_size': len(self.loader.partition),
            'embedding': z.detach().cpu().numpy()
        }
        
    @torch.no_grad()
    def validation_step(self, x_batch, y_batch):
        self.encoder.eval()
        self.head.eval()
        y_hat = self.head(self.encoder(x_batch))
        lss = F.cross_entropy(y_hat, y_batch)
        return y_hat, lss.item()

    @torch.no_grad()
    def get_n_params(self, model):
        u,v,m = 0,0,0 
        for layer, params in model.state_dict().items():
            if len(params.size()) > 0:
                if 'mask' in layer:
                    params = params[torch.where(torch.abs(self.prune(params))>=self.args.lambda_l1)]
                    nn=1
                    for s in list(params.size()):
                        nn *= s
                    m += nn
                elif 'uu' in layer:
                    nn=1
                    for s in list(params.size()):
                        nn *= s
                    u += nn
                elif 'vv' in layer:
                    nn=1
                    for s in list(params.size()):
                        nn *= s
                    v += nn
        return u,v,m
    
    def prune(self, mask):
        pruned = torch.abs(mask) < self.args.lambda_l1
        return mask.masked_fill(pruned, 0)









    
    
