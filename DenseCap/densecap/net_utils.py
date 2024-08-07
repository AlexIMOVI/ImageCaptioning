import torch
from torch.utils.model_zoo import load_url
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def load_cnn(name, backend='nn', path_offset=None):
    if name == 'vgg-16':
        model = torchvision.models.vgg16(weights=torchvision.models.vgg.VGG16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f'Unrecognized model "{name}"')
    return model


cudnn_algos = {
    'vgg-16': {
        'NVIDIA GeForce RTX 2080 Ti': {
            1: (1, 0, 0),
            3: (1, 1, 0),
            6: (1, 1, 3),
            8: (1, 1, 3),
            11: (1, 1, 3),
            13: (1, 1, 3),
            15: (1, 1, 3),
            18: (1, 1, 3),
            20: (1, 1, 0),
            22: (1, 1, 0),
            25: (1, 1, 0),
            27: (1, 1, 0),
            29: (1, 1, 0),
        }
    }
}


def cudnn_tune_cnn(cnn_name, cnn):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        if cnn_name in cudnn_algos and device_name in cudnn_algos[cnn_name]:
            algos = cudnn_algos[cnn_name][device_name]
            for i, layer in enumerate(cnn.features, 1):
                if isinstance(layer, nn.Conv2d) and i in algos:
                    algo = algos[i]
                    layer = nn.Conv2d(layer.in_channels, layer.out_channels,
                                      layer.kernel_size, layer.stride,
                                      layer.padding, layer.dilation,
                                      layer.groups, layer.bias is not None)
                    cnn.features[i - 1] = layer
    return cnn


def subsequence(net, start_idx, end_idx):
    return net.features[start_idx: end_idx]


def compute_field_centers(net, end_idx=None):
    end_idx = end_idx or len(net)
    x0, y0 = 0, 0
    sx, sy = 1, 1
    for i, layer in enumerate(net, 1):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv2d) or \
           isinstance(layer, nn.Conv2d):
            unit_stride = layer.stride == (1, 1)
            same_x = (layer.kernel_size[0] // 2) == layer.padding[0]
            same_y = (layer.kernel_size[1] // 2) == layer.padding[1]
            same_conv = unit_stride and same_x and same_y
            if not same_conv:
                raise ValueError('Cannot handle this type of conv layer')
        elif isinstance(layer, nn.ReLU):
            pass  # Do nothing
        elif isinstance(layer, nn.MaxPool2d):
            if layer.kernel_size != 2 or layer.stride != 2:
                raise ValueError('Cannot handle this type of pooling layer')
            x0 = x0 + sx / 2
            y0 = y0 + sy / 2
            sx = sx*2
            sy = sy*2
        else:
            raise ValueError(f'Cannot handle layer of type {type(layer)}')
    return x0, y0, sx, sy


def get_named_modules(gmod):
    name_to_mods = {}
    for node in gmod.forwardnodes:
        if node.data.module:
            node_name = node.data.annotations['name']
            if node_name:
                assert node_name not in name_to_mods, 'Node names must be unique'
                name_to_mods[node_name] = node.data.module
    return name_to_mods


def display_loss_history(loss_dict, loss_name, model_name, pad, save_fig=True):
    total = []
    for d in loss_dict:
        total.append(d[loss_name].cpu().detach().numpy())
    r = np.arange(0, pad*len(loss_dict), pad)
    plt.plot(r[(len(loss_dict)//10):], total[(len(loss_dict)//10):])
    plt.xlabel('iters')
    plt.title(loss_name)
    if save_fig:
        plt.savefig(f'graphs/{loss_name}_{model_name}{pad}k')
    plt.show()



