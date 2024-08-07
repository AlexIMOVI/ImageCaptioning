import json
from collections import Counter, OrderedDict
from itertools import chain
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
mypath = 'AlexCap/models_pth'
onlyfiles = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]
for path in onlyfiles:
    dic = torch.load(path)
    dic2 = OrderedDict()
    for key in list(dic.keys()):
        if 'resnet_backbone' not in key:
            dic2[key] = dic[key]
    torch.save(dic2, path)

INPUT_TRAIN = 'AlexCap/logs/loss_history_celeba_vggface_lstm_ft_bs1_rand.json'
INPUT_VAL = 'AlexCap/face2text/clean_dev_2.1.json'
INPUT_TEST = 'AlexCap/face2text/clean_test_2.1.json'
attr_file = 'AlexCap/face2text/list_attr_celeba.csv'
train = json.load(open(INPUT_TRAIN, 'r'))
val = json.load(open(INPUT_VAL, 'r'))
test = json.load(open(INPUT_TEST, 'r'))
data_list = list(chain(*[train, val, test]))
name_list = list(dic['filename'] for dic in data_list)
attr_csv = pd.read_csv(attr_file, index_col='image_id')
attr_label = list(attr_csv.columns)
idx_list = list(attr_csv.index)
idx_array = list(idx_list.index(dic['filename']) for dic in data_list)
attr_list = attr_csv.values[idx_array]
last_dic = test[-1]
del_idx = []
for i, dic in enumerate(test):
    dic['description'] = [dic['description']]
    if dic['filename'] == last_dic['filename']:
        for s in last_dic['description']:
            dic['description'].append(s)
        del_idx.append(i-1)
    last_dic = dic
res = [item for i, item in enumerate(test) if i not in del_idx]
for dic in res:
    if len(dic['description']) > 1:
        lengths = [len(seq) for seq in dic['description']]
        dic['description'] = [dic['description'][lengths.index(max(lengths))]]
with open('AlexCap/face2text/my_clean_test_2.1.json', 'w') as f:
    json.dump(res, f)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
class CustomImageFolder:
    def __init__(self, root, transform=None, index_list=None):
        self.index_list = index_list if index_list is not None else []
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root)
                            if fname.endswith('.jpg') and fname in self.index_list]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Get the original item
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the dataset using the custom class
dataset = CustomImageFolder(root='AlexCap/face2text/img_align_celeba/img_align_celeba', transform=transform, index_list=name_list)

# Initialize variables to store sum and sum of squares
num_samples = 0
channel_sum = torch.zeros(3)
channel_sum_sq = torch.zeros(3)
length = dataset.__len__()
# Iterate over the dataset
for i in range(length):
    # Filter out None values returned for ignored images
    image = dataset.__getitem__(i)

    num_samples += 1
    channel_sum += image.sum(dim=[1, 2]) / (image.size(1) * image.size(2))
    channel_sum_sq += torch.sqrt(((image ** 2).sum(dim=[1, 2]) / (image.size(1) * image.size(2))) - ((image.sum(dim=[1, 2]) / (image.size(1) * image.size(2)))**2))

# Calculate mean and std
mean = channel_sum / num_samples
std = channel_sum_sq / num_samples

print(f"Mean: {mean}")
print(f"Std: {std}")

import matplotlib.pyplot as plt
# input_json = 'logs/results_history_transformer_gt_normalized.json'
# train_json = 'logs/loss_history_transformer_gt_normalized.json'
# info = json.load(open(input_json, 'r'))
# train_loss = json.load(open(train_json, 'r'))
#
# # losses = [o['loss_results'] for o in info]
# # ap = [o['ap_results']['map']*100 for o in info]
# # step = info[0]['best_iter'] + 1
# # steps = np.arange(step, len(info) * step + 1, step)
# # t_loss = [o['captioning_loss'] for o in train_loss][:steps[-1]//500+1]
# # t_steps = np.arange(0, len(info) * step + 1, 500)
# # fig, ax = plt.subplots(2, 1, sharex='col')
# # ax[0].plot(steps, losses, 'bo-')
# # ax[0].plot(t_steps, t_loss, 'g--')
# # ax[0].set_ylabel('loss')
# # ax[0].set_title('Loss and Average Precision (AP) during training, on evaluation dataset')
# # ax[1].plot(steps, ap, 'ro-')
# # ax[1].set_ylabel('mAP (%)')
# # fig.text(.5, .04, 'iter')
# # plt.show()
