import os
import time
import numpy as np
from torchvision import datasets
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from misc.utils import *

data_path = ''
seed = 1234
ratio_train = 0.8
alpha = 0.5

def generate_data(dataset, n_clients):
    st = time.time()
    x, y, n_clss = get_data(dataset)
    x, y = split_train(x, y, dataset)
    x_clss = {}
    y_clss = {}
    for c_id in np.arange(n_clss):
        idx = np.where(np.array(y)==c_id)[0] 
        x_clss[c_id] = [x[i] for i in idx]
        y_clss[c_id] = [y[i] for i in idx]
    split_iid(x_clss, y_clss, n_clients, dataset)
    split_non_iid(x_clss, y_clss, n_clients, n_clss, dataset)
    print(f'done ({time.time()-st:.2f})')

def get_data(dataset):
    st = time.time()
    if dataset == 'CIFAR_100':
        data = [datasets.CIFAR100(data_path, train=True, download=True),
                    datasets.CIFAR100(data_path, train=False, download=True)]
    elif dataset == 'CIFAR_10':
        data = [datasets.CIFAR10(data_path, train=True, download=True),
                    datasets.CIFAR10(data_path, train=False, download=True)]
    elif dataset == 'SVHN':
        data = [datasets.SVHN(data_path, split='train', download=True),
                    datasets.SVHN(data_path, split='test', download=True)]
    elif dataset == 'MNIST':
        data = [datasets.MNIST(data_path, train=True, download=True),
                    datasets.MNIST(data_path, train=False, download=True)]

    print(f'{dataset} have been loaded ({time.time()-st:.2f} sec)')
    x, y = [], []
    for d in data:
        for image, target in iter(d):
            x.append(image)
            y.append(target)
    x, y = shuffle(seed, x, y)
    return x, y, len(set(y))

def split_train(x, y, dataset):
    st=time.time()
    n_data = len(x)
    ratio_test = (1-ratio_train)/2
    n_train = round(len(x) * ratio_train)
    n_test = round(len(x) * ratio_test)
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:n_train+n_test]
    y_test = y[n_train:n_train+n_test]
    x_val = x[n_train+n_test:]
    y_val = y[n_train+n_test:]
    torch_save(data_path,f'{dataset}_train.pt', {
        'x': x_train,
        'y': y_train,
    })
    torch_save(data_path,f'{dataset}_test.pt', {
        'x': x_test,
        'y': y_test,
    })
    torch_save(data_path,f'{dataset}_val.pt', {
        'x': x_val,
        'y': y_val,
    })
    print(f'splition done, n_train:{n_train}, n_test:{n_test}, n_val:{len(x_val)} ({time.time()-st:.2f} sec)')
    return x_train, y_train

def split_iid(x_clss, y_clss, n_clients, dataset):
    st = time.time()
    n_inst_per_clss = {c_id:round(len(data)/n_clients) for c_id, data in x_clss.items()} 
    for client_id in range(n_clients):
        x_part, y_part = [], []
        for clss_id in x_clss.keys():
            x_part = [*x_part, *x_clss[clss_id][client_id*n_inst_per_clss[clss_id]:(client_id+1)*n_inst_per_clss[clss_id]]]
            y_part = [*y_part, *y_clss[clss_id][client_id*n_inst_per_clss[clss_id]:(client_id+1)*n_inst_per_clss[clss_id]]]
        x_part, y_part = shuffle(seed, x_part, y_part)
        torch_save(data_path,f'{dataset}_iid_partition_{client_id}.pt', {
            'x': x_part,
            'y': y_part,
            'client_id': client_id
        })
        print(f'client_id:{client_id}, iid, n_train:{len(x_part)} ({time.time()-st:.2f})')
        st = time.time()

def split_non_iid(x_clss, y_clss, n_clients, n_clss, dataset):
    st = time.time()
    dist = np.random.dirichlet([alpha for _ in range(n_clients)], n_clss)
    n_data_per_clss = [len(x) for c, x in x_clss.items()]
    for client_id in range(n_clients):
        x_part = []
        y_part = []
        _n_data_per_clss = []
        for i, clss_id in enumerate(x_clss.keys()):
            _n = int(n_data_per_clss[clss_id] * dist[clss_id][client_id])
            _n_data_per_clss.append(_n)
            x_part = [*x_part, *x_clss[clss_id][:_n]]
            x_clss[clss_id] = x_clss[clss_id][_n:]
            _y = [clss_id]*_n
            y_part = [*y_part, *_y]
        x_part, y_part = shuffle(seed, x_part, y_part)
        torch_save(data_path,f'{dataset}_non_iid_partition_{client_id}.pt', {
            'x': x_part,
            'y': y_part,
            'client_id': client_id
        })
        print(f'client_id:{client_id}, non_iid, n_train:{len(x_part)}, n_data_per_clss:{_n_data_per_clss}, ({time.time()-st:.2f})')
        st = time.time()

def generate_cifar_100_domain(dataset='CIFAR_100'):
    # laod cifar 10
    x, y, n_clss = get_data(dataset)
    # specify domains
    classes, names = cifar100_superclass_label_pair()
    domains = [{n:classes[i][j] for j,n in enumerate(ns)} for i,ns in enumerate(names)]
    # split domains
    for dom_id, dom in enumerate(domains):
        st = time.time()
        x_dom, y_dom, new_c_id, dom_info = [], [] , 0, {}
        for c_name, c_id in dom.items():
            idx = np.where(np.array(y)==c_id)[0] 
            x_dom = [*x_dom, *[x[i] for i in idx]]
            y_dom = [*y_dom, *([new_c_id]*len(idx))]
            dom_info[new_c_id] = c_name
            new_c_id += 1
        x_dom, y_dom = shuffle(seed, x_dom, y_dom)
        # split tr, te
        n_train = round(len(x_dom) * 0.8)
        n_test = round(len(x_dom) * 0.1)
        x_train = x_dom[:n_train]
        y_train = y_dom[:n_train]
        x_test = x_dom[n_train:n_train+n_test]
        y_test = y_dom[n_train:n_train+n_test]
        x_val = x_dom[n_train+n_test:]
        y_val = y_dom[n_train+n_test:]
        torch_save(data_path,f'CIFAR_100_dom_{dom_id}_partition.pt', {
            'x': x_train,
            'y': y_train,
            'client_id': dom_id,
            'info': dom_info,
        })
        torch_save(data_path,f'CIFAR_100_dom_{dom_id}_test.pt', {
            'x': x_test,
            'y': y_test,
            'client_id': dom_id,
            'info': dom_info
        })
        torch_save(data_path,f'CIFAR_100_dom_{dom_id}_val.pt', {
            'x': x_val,
            'y': y_val,
            'client_id': dom_id,
            'info': dom_info
        })
        print(f'dom_id:{dom_id}, n_train:{len(x_train)}, n_test:{len(x_test)}, n_val:{len(x_val)} ({time.time()-st})')

def cifar100_superclass_label_pair():
    CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    sclass = []
    sclass.append(['beaver', 'dolphin', 'otter', 'seal', 'whale'])                      #aquatic mammals
    sclass.append(['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'])               #fish
    sclass.append(['orchid', 'poppy', 'rose', 'sunflower', 'tulip'])                    #flowers
    sclass.append(['bottle', 'bowl', 'can', 'cup', 'plate'])                            #food
    sclass.append(['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'])              #fruit and vegetables
    sclass.append(['clock', 'keyboard', 'lamp', 'telephone', 'television'])             #household electrical devices
    sclass.append(['bed', 'chair', 'couch', 'table', 'wardrobe'])                       #household furniture
    sclass.append(['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'])           #insects
    sclass.append(['bear', 'leopard', 'lion', 'tiger', 'wolf'])                         #large carnivores
    sclass.append(['bridge', 'castle', 'house', 'road', 'skyscraper'])                  #large man-made outdoor things
    sclass.append(['cloud', 'forest', 'mountain', 'plain', 'sea'])                      #large natural outdoor scenes
    sclass.append(['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'])            #large omnivores and herbivores
    sclass.append(['fox', 'porcupine', 'possum', 'raccoon', 'skunk'])                   #medium-sized mammals
    sclass.append(['crab', 'lobster', 'snail', 'spider', 'worm'])                       #non-insect invertebrates
    sclass.append(['baby', 'boy', 'girl', 'man', 'woman'])                              #people
    sclass.append(['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'])               #reptiles
    sclass.append(['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'])                  #small mammals
    sclass.append(['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'])  #trees
    sclass.append(['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'])            #vehicles 1
    sclass.append(['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'])             #vehicles 2
    
    labels_pair = [[cid for cid in range(100) if CIFAR100_LABELS_LIST[cid] in sclass[gid]] for gid in range(20)]

    return labels_pair, sclass

generate_data(dataset='CIFAR_100', n_clients=20)
generate_data(dataset='CIFAR_10', n_clients=20)
generate_data(dataset='SVHN', n_clients=20)
