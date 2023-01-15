import pdb
import time
import torch
import numpy as np

from modules.federated import ServerModule
from modules.nets import *
from misc.utils import *

class Server(ServerModule):

    def __init__(self, args, sd, gpus):
        super(Server, self).__init__(args, sd, gpus)
        self.encoder = ResNet9(factorize=True, l1=self.args.lambda_l1, rank=self.args.rank).cpu()
        self.head = Head(self.args.n_clss, factorize=True, l1=self.args.lambda_l1, rank=self.args.rank).cpu()

        self.log = {}
        self.matching = {cid:{c:0 for c in range(self.args.n_clients)} for cid in range(self.args.n_clients)}
        self.cid_to_w = {}
        self.cid_to_z = {}
        self.cid_to_tr_size = {}
        self.criteria_noise = torch_load(self.args.data_path, 'noise.pt')
        
    def on_round_begin(self, selected, curr_rnd):
        self.curr_rnd = curr_rnd
        if self.curr_rnd == 0:
            self.sd['noise'] = self.criteria_noise
            self.sd['init'] = {'encoder': get_state_dict(self.encoder),
                                    'head': get_state_dict(self.head)}
        else:
            for cid in selected:
                self.sd[f'{cid}_encoder'] = self.get_weights(cid, curr_rnd)

    def on_round_complete(self, updated):
        self.update(updated)
        self.log[self.curr_rnd] = self.matching.copy()
        self.save_log()
    
    def update(self, updated):
        st = time.time()
        for cid in updated:
            loaded = self.sd[cid].copy()
            self.cid_to_w[cid] = loaded['encoder']
            self.cid_to_z[cid] = loaded['embedding']
            self.cid_to_tr_size[cid] = loaded['train_size']
            del self.sd[cid]

        self.cid_to_v = {}
        for cid, w in self.cid_to_w.items():
            self.cid_to_v[cid] = self.linearize(w)
        print(f'[server] all clients updated ({time.time()-st:.2f} s)')

    def save_log(self):
        try:
            save(self.args.log_path, f'server.txt', {
                'log': self.log
            })
        except:
            pdb.set_trace()

    def get_weights(self, client_id, curr_rnd):
        st = time.time()
        if self.curr_rnd == 0:
            with torch.no_grad():
                theta = {}
                for name, params in self.encoder.named_parameters():
                    theta[name] = params.data.clone().detach().cpu().numpy()
                return theta
        else:
            if not client_id in self.cid_to_z:
                theta_list = [theta for cid, theta in self.cid_to_w.items()]
                tr_size_list = [self.cid_to_tr_size[cid] for cid, _ in self.cid_to_w.items()]
                ratio = (np.array(tr_size_list)/np.sum(tr_size_list)).tolist()
                return self.aggregate(theta_list, ratio)
            else:
                # Best                
                cids, sigma = self.calculate_similarity(self.cid_to_v[client_id], self.cid_to_v)
                n_aggr = torch.sum(torch.where(sigma>=self.args.tau, 1, 0)).item()
                n_aggr = n_aggr if n_aggr > self.args.minimum_aggr else self.args.minimum_aggr
                cids = cids[:n_aggr]
                scores = np.exp(sigma[:n_aggr].numpy() * self.args.alpha) 
                omega = scores/np.sum(scores)
                print(f'[c:{client_id}], n_aggr:{n_aggr}, reflection:{[(round(cids[i],5),round(sigma[i].item(),5),round(om,5)) for i,om in enumerate(omega)]}')

                for cid in cids:
                    self.matching[client_id][cid] += 1

                local_model = self.cid_to_w[client_id]
                for name, params in local_model.items():
                    if self.args.aggr == 'u':
                        if 'mask' in name or 'vv' in name:
                            print(f'aggr:{self.args.aggr},  {name} skipped')
                            continue
                    elif self.args.aggr == 'um':
                        if  'vv' in name:
                            print(f'aggr:{self.args.aggr},  {name} skipped')
                            continue
                    elif self.args.aggr == 'uv':        
                        if  'mask' in name:
                            print(f'aggr:{self.args.aggr},  {name} skipped')
                            continue
                    elif self.args.aggr == 'uvm':
                        pass
                    
                    # local_model[name] = params + np.sum([omega[1:][i]*(self.cid_to_w[cid][name]-params) for i,cid in enumerate(cids[1:])], 0)
                    local_model[name] = np.sum([self.cid_to_w[c_id][name]*omega[i] for i, c_id in enumerate(cids)], 0)
                return local_model

    def calculate_similarity(self, z, z_list):
        z = np.expand_dims(z, 0)
        zs = np.array([emb for emb in z_list.values()])
        cids = np.array([cid for cid in z_list.keys()])
        similarity = F.cosine_similarity(torch.tensor(z).view(1,-1), torch.tensor(zs).squeeze())
        scores, sorted_idx = torch.sort(similarity, 0, descending=True)
        sorted_cid = [cids[i] for i in sorted_idx.numpy()]
        return sorted_cid, scores

    def linearize(self, state_dict):
        params = []
        for name, param in state_dict.items():
            if 'encoder.6' in name and 'vv' in name:
                params.append(param.reshape(-1))
        return np.concatenate(params, 0)



