import h5py
import torch
import json
from AlexCap.densecap_utils import utils
import torchvision
from torchvision import transforms
import numpy as np

class AlexDataLoader:
    def __init__(self, opt):

        self.h5_file_path = utils.getopt(opt, 'data_h5',
                                         'AlexCap/data/face2text-data.h5')  # Required HDF5 file with images and other data
        self.json_file_path = utils.getopt(opt, 'data_json', 'AlexCap/data/face2text-dicts.json')
        self.device = opt['device']
        with open(self.json_file_path, 'r') as f:
            self.info = json.load(f)
        self.vocab_size = len(self.info['idx_to_token'])
        self.idx_to_token = []
        for k in self.info['idx_to_token'].items():
            self.idx_to_token.append(k)

        print(f'DataLoader loading h5 file: {self.h5_file_path}')
        self.h5_file = h5py.File(self.h5_file_path, 'r')

        keys = ['img_to_first_phr', 'img_to_last_phr', 'labels', 'lengths', 'split', 'attributes']
        for key in keys:
            print(f'reading {key}')
            setattr(self, key, self.h5_file['/'+key][:])

        images_size = self.h5_file['images'].shape
        assert len(images_size) == 4, '/images should be a 4D tensor'
        self.num_images = images_size[0]
        self.num_channels = images_size[3]
        self.seq_length = self.h5_file['labels'].shape[1]
        self.iterators = {0: 0, 1: 0, 2: 0}
        self.attributes_labels = self.info['attributes_labels']
        self.preprocess = torchvision.models.ResNet101_Weights.IMAGENET1K_V2.transforms()
        self.train_ix, self.val_ix, self.test_ix = [], [], []
        for i in range(self.num_images):
            if self.h5_file['split'][i] == 0:
                self.train_ix.append(i)
            elif self.h5_file['split'][i] == 1:
                self.val_ix.append(i)
            elif self.h5_file['split'][i] == 2:
                self.test_ix.append(i)
    def getSeqLength(self):
        return self.seq_length

    def getVocabSize(self):
        return self.vocab_size

    def reset_iterator(self, split_val):
        self.iterators[split_val] = 0

    def get_batch(self, opt, batch_size, idx=-1):
        split_val = utils.getopt(opt, 'split', 0)
        iterate = utils.getopt(opt, 'iterate', True)
        split_ix = []
        assert split_val == 0 or split_val == 1 or split_val == 2,\
            'split must be integer, either 0 (train), 1 (val) or 2 (test)'
        if split_val == 0:
            split_ix = self.train_ix
        if split_val == 1:
            split_ix = self.val_ix
        if split_val == 2:
            split_ix = self.test_ix
        assert len(split_ix) > 0, 'split is empty ?'
        max_index = len(split_ix)
        if iterate:
            ri = self.iterators[split_val]
            ri_next = ri + batch_size
            if ri_next >= max_index:
                ri_next = 0
            self.iterators[split_val] = ri_next
            ix = split_ix[ri:ri+batch_size]
        else:
            if idx == -1:
                ri = np.sort(torch.multinomial(torch.ones(max_index), batch_size).cpu().numpy())
                ix = [split_ix[r] for r in ri]
            else:
                ri = idx

        img = torch.from_numpy(self.h5_file['/images'][ix, :, :, :self.num_channels]).permute(0,3,1,2)
        img = self.preprocess(img)
        label_array = torch.from_numpy(self.labels[ix])
        attr_array = torch.clamp(torch.from_numpy(self.attributes[ix]), 0)
        filename = [self.info['idx_to_filename'][str(i)] for i in ix]
        assert filename is not None
        info_table = [{"filename": filename,
                       "split_bounds": [ri, max_index]
                       }]

        return img.to(self.device), label_array.to(self.device), info_table, attr_array
