import os
import pdb
import glob
import json
import time
import torch
import random
import numpy as np
from collections import OrderedDict
from misc.forked_pdb import ForkedPdb

def str2bool(v):
  return v.lower() in ['true', 't']

def torch_save(base_dir, filename, data):
    # if os.path.isdir(base_dir) == False:
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)
    # print('file saved ({})'.format(fpath))

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))

def shuffle(seed, x, y):
    idx = np.arange(len(x))
    random.seed(seed)
    random.shuffle(idx)
    return [x[i] for i in idx], [y[i] for i in idx]

def save(base_dir, filename, data):
    # if os.path.isdir(base_dir) == False:
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, filename), 'w+') as outfile:
        json.dump(data, outfile)

def exists(base_dir, filename):
    return os.path.exists(os.path.join(base_dir, filename))

def join_glob(base_dir, filename):
    return glob.glob(os.path.join(base_dir, filename))

def remove_if_exist(base_dir, filename):
    targets = join_glob(base_dir, filename)
    if len(targets)>0:
        for t in targets:
            os.remove(t)

def debugger():
    ForkedPdb().set_trace()

def get_state_dict(model):
    state_dict = convert_tensor_to_np(model.state_dict())
    return state_dict

def set_state_dict(model, state_dict, gpu_id, skip=False, skip_bn=False, include=False):
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip=skip, skip_bn=skip_bn, include=include, model=model.state_dict())
    model.load_state_dict(state_dict)
    
def convert_tensor_to_np(state_dict):
    return OrderedDict([(k,v.clone().detach().cpu().numpy()) for k,v in state_dict.items()])

def convert_np_to_tensor(state_dict, gpu_id, skip=False, skip_bn=False, include=False, model=None):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        if skip_bn:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue

        if not isinstance(include, bool):
            if include in k:
                pass 
            else:
                _state_dict[k] = model[k]
                continue

        if not isinstance(skip, bool):
            if skip in k:
                # print(k, 'skipped')
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = torch.tensor(v).cuda(gpu_id)
        else:
            _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id)
    return _state_dict

def convert_np_to_tensor_cpu(state_dict):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        _state_dict[k] = torch.tensor(v)
    return _state_dict
