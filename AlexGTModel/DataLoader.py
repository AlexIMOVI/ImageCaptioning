import h5py
import torch
import json
from PIL import Image
import torchvision.io

from AlexGTModel.densecap_utils import utils
from torchvision import transforms

class DataLoader:
    def __init__(self, opt):
        self.h5_file_path = utils.getopt(opt, 'data_h5', 'data/VG-regions.h5')  # Required HDF5 file with images and other data
        self.json_file_path = utils.getopt(opt,'data_json', 'data/VG-regions-dicts.json')  # Required JSON file with vocab, etc.
        self.debug_max_train_images = utils.getopt(opt,'debug_max_train_images', -1)
        self.proposal_regions_h5 = utils.getopt(opt,'proposal_regions_h5', '')
        self.device = opt['device']
        # Load the JSON file which contains additional information about the dataset
        print(f'DataLoader loading json file: {self.json_file_path}')
        with open(self.json_file_path, 'r') as f:
            self.info = json.load(f)
        self.vocab_size = len(self.info['idx_to_token'])

        # Convert keys in idx_to_token from string to integer
        idx_to_token = []
        for k in self.info['idx_to_token'].items():
            idx_to_token.append(k)
        # self.idx_to_token = {int(k): v for k, v in self.info['idx_to_token'].items()}
        self.idx_to_token = idx_to_token
        # Open the HDF5 file
        print(f'DataLoader loading h5 file: {self.h5_file_path}')
        self.h5_file = h5py.File(self.h5_file_path, 'r')
        # Load datasets from the HDF5 file
        keys = ['box_to_img', 'boxes', 'image_heights', 'image_widths', 'img_to_first_box', 'img_to_last_box',
                'labels', 'lengths', 'original_heights', 'original_widths', 'split']
        for key in keys:
            print(f'reading {key}')
            setattr(self, key, self.h5_file['/'+key][:])
        # Open region proposals file for reading, if specified
        if len(self.proposal_regions_h5)>0:
            print(f'DataLoader loading objectness boxes from h5 file: {self.proposal_regions_h5}')
            self.obj_boxes_file = h5py.File(self.proposal_regions_h5, 'r')
            self.obj_img_to_first_box = self.obj_boxes_file['img_to_first_box'][:]
            self.obj_img_to_last_box = self.obj_boxes_file['img_to_last_box'][:]
        else:
            self.obj_boxes_file = None
        # Extract image size from dataset
        images_size = self.h5_file['images'].shape
        assert len(images_size) == 4, '/images should be a 4D tensor'
        assert images_size[1] == images_size[2], 'width and height must match'
        self.num_images = images_size[0]
        self.num_channels = images_size[3]
        self.max_image_size = images_size[2]

        # Extract some attributes from the data
        self.num_regions = self.h5_file['boxes'].shape[0]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.seq_length = self.h5_file['labels'].shape[1]

        # Set up index ranges for the different splits
        self.train_ix, self.val_ix, self.test_ix = [], [], []
        for i in range(self.num_images):
            if self.h5_file['split'][i] == 0:
                self.train_ix.append(i)
            elif self.h5_file['split'][i] == 1:
                self.val_ix.append(i)
            elif self.h5_file['split'][i] == 2:
                self.test_ix.append(i)

        self.iterators = {0: 0, 1: 0, 2: 0}  # Iterators for train/val/test
        print(f'assigned {len(self.train_ix)}/{len(self.val_ix)}/{len(self.test_ix)} images to train/val/test.')

        print('initialized DataLoader:')
        print(f'#images: {self.num_images}, #regions: {self.num_regions}, sequence max length: {self.seq_length}')

    def getImageMaxSize(self):
        return self.max_image_size

    def getSeqLength(self):
        return self.seq_length

    def getVocabSize(self):
        return self.vocab_size

    def getVocab(self):
        return self.info['idx_to_token']

    def decodeSequence(self,seq):
        D, N = seq.shape[0], seq.shape[1]
        out = []
        itow = self.info['idx_to_token']
        for i in range(N):
            txt = []
            for j in range(D):
                ix = seq[j,i].items()
                if (1 <= ix <= self.vocab_size):
                    if (j>=1):
                        txt.append(' ')
                    txt.append(itow[str(ix)])
                else:
                    break
            out.append(''.join(txt))
        return out

    def reset_iterator(self, split):
        assert split in [0, 1, 2]
        self.iterators[split] = 0

    def get_batch(self, opt, idx=-1):
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
        if self.debug_max_train_images > 0:
            max_index = self.debug_max_train_images
        if iterate:
            ri = self.iterators[split_val]
            ri_next = ri+1
            if ri_next >= max_index:
                ri_next = 0
            self.iterators[split_val] = ri_next
        else:
            if idx == -1:
                ri = torch.randint(max_index, size=[1])
            else:
                ri = idx

        ix = split_ix[ri]
        assert ix is not None, 'bug ix faux'

        img = transforms.ToTensor()(self.h5_file['/images'][ix, :self.max_image_size, :self.max_image_size, :self.num_channels])
        img = img[:, :self.image_heights[ix], :self.image_widths[ix]]
        img = img.float()
        img = img.unsqueeze(0).to(self.device)
        img = self.normalize(img)

        r0 = self.img_to_first_box[ix]
        r1 = self.img_to_last_box[ix]
        label_array = torch.from_numpy(self.labels[r0-1:r1])
        box_batch = torch.from_numpy(self.boxes[r0-1:r1])

        assert label_array.dim() == 2
        assert box_batch.dim() == 2
        label_array = label_array.view(1, label_array.size(0), label_array.size(1))
        box_batch = box_batch.view(1, box_batch.size(0), box_batch.size(1))

        filename = self.info['idx_to_filename'][str(ix+1)]
        assert filename is not None
        w, h = self.image_widths[ix], self.image_heights[ix]
        ow, oh = self.original_widths[ix], self.original_heights[ix]
        info_table = [{"filename": filename,
                       "split_bounds": [ri+1, len(split_ix)],
                       "width": w, "height": h,
                       "ori_width": ow, "ori_height": oh}]

        return img.to(self.device), box_batch, label_array, info_table


class ImageProcessor:
    def __init__(self, opt):
        self.device = opt.device
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.max_size = opt.max_img_size
        self.resize = torchvision.transforms.Resize(700, max_size=720)
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
    def preprocess_img(self, img_path):
        yolo_img = Image.open(img_path)
        res = self.yolo_model(yolo_img)
        img = self.resize(torchvision.io.read_image(img_path))
        img = img.float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        img = self.normalize(img)
        gt_boxes = res.xywh[0][:,:4].unsqueeze(0)
        return img, gt_boxes

