import torch
from torch import nn
import torch.nn.functional as F
import pickle
import mxnet as mx
from mxnet import ndarray as nd


'''
Forked from InsightFace (https://github.com/deepinsight/insightface)
'''

# Loading datasets
@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list

# You run this only once!
@torch.no_grad()
def feat_ext(dataset, backbone, batch_size, device):
    data, issame = dataset[0], dataset[1]
    len_data = data[0].size(0)
    feats = []
    for dset in data:
        tmp = []
        ba = 0
        while ba < len_data:
            bb = min(ba + batch_size, len_data)
            img = dset[ba:bb]
            img = (img - 127.5) / 127.5
            feat = F.normalize(backbone(img.to(device)))
            tmp.append(feat)
            ba = bb
        tmp = torch.cat(tmp, dim = 0)
        feats.append(tmp)
        
    feats = F.normalize(feats[0] + feats[1])
    return feats


