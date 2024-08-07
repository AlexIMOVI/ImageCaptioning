

# coding=utf8

import argparse, os, json, string
from collections import Counter
from queue import Queue
from threading import Thread, Lock

from math import floor
import h5py
import numpy as np
from imageio.v2 import imread
from PIL import Image
from itertools import chain
import pandas as pd

"""
This file expects a JSON file containing ground-truth regions and captions
in the same format as the region descriptions file from the Visual Genome
website. Concretely, this is a single large JSON file containing a list;
each element of the list describes a single image and has the following
format:

{
  "id": [int], Unique identifier for this image,
  "regions": [
    {
      "id": [int] Unique identifier for this region,
      "image": [int] ID of the image to which this region belongs,
      "height": [int] Height of the region in pixels,
      "width": [int] Width of the region in pixels,
      "phrase": [string] Caption for this region,
      "x": [int] x-coordinate of the upper-left corner of the region,
      "y": [int] y-coordinate of the upper-left corner of the region,
    },
    ...
  ]
}

We assume that all images are on disk in a single folder, and that
the filename for each image is the same as its id with a .jpg extension.

This file will be preprocessed into an HDF5 file and a JSON file with
some auxiliary information. The captions will be tokenized with some
basic preprocessing (split by words, remove special characters).

Note, in general any indices anywhere in input/output of this file are 1-indexed.

The output JSON file is an object with the following elements:
- token_to_idx: Dictionary mapping strings to integers for encoding tokens, 
                in 1-indexed format.
- filename_to_idx: Dictionary mapping string filenames to indices.
- idx_to_token: Inverse of the above.
- idx_to_filename: Inverse of the above.

The output HDF5 file has the following format to describe N images with
M total regions:

- images: uint8 array of shape (N, 3, image_size, image_size) of pixel data,
  in BDHW format. Images will be resized so their longest edge is image_size
  pixels long, aligned to the upper left corner, and padded with zeros.
  The actual size of each image is stored in the image_heights and image_widths
  fields.
- image_heights: int32 array of shape (N,) giving the height of each image.
- image_widths: int32 array of shape (N,) giving the width of each image.
- original_heights: int32 array of shape (N,) giving the original height of
  each image.
- original_widths: int32 array of shape (N,) giving the original width of
  each image.
- boxes: int32 array of shape (M, 4) giving the coordinates of each bounding box.
  Each row is (xc, yc, w, h) where yc and xc are center coordinates of the box,
  and are one-indexed.
- lengths: int32 array of shape (M,) giving lengths of label sequence for each box
- captions: int32 array of shape (M, L) giving the captions for each region.
  Captions in the input with more than L = --max_token_length tokens are
  discarded. To recover a token from an integer in this matrix,
  use idx_to_token from the JSON output file. Padded with zeros.
- img_to_first_box: int32 array of shape (N,). If img_to_first_box[i] = j then
  captions[j] and boxes[j] give the first annotation for image i
  (using one-indexing).
- img_to_last_box: int32 array of shape (N,). If img_to_last_box[i] = j then
  captions[j] and boxes[j] give the last annotation for image i
  (using one-indexing).
- box_to_img: int32 array of shape (M,). If box_to_img[i] = j then then
  regions[i] and captions[i] refer to images[j] (using one-indexing).
"""


def build_vocab(data, min_token_instances, verbose=True):
    """ Builds a set that contains the vocab. Filters infrequent tokens. """
    token_counter = Counter()
    for img in data:
        for phrase in img['description']:
            if phrase is not None:
                token_counter.update(phrase)
    vocab = set()
    for token, count in token_counter.items():
        if count >= min_token_instances:
            vocab.add(token)
    if verbose:
        print('Keeping %d / %d tokens with enough instances'
              % (len(vocab), len(token_counter)))

    if len(vocab) < len(token_counter):
        vocab.add('<UNK>')
        if verbose:
            print('adding special <UNK> token.')
    else:
        if verbose:
            print('no <UNK> token needed.')
    return vocab


def build_vocab_dict(vocab):
    token_to_idx, idx_to_token = {}, {}
    next_idx = 1

    for token in vocab:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def encode_caption(tokens, token_to_idx, max_token_length):
    encoded = np.zeros(max_token_length, dtype=np.int32)
    for i, token in enumerate(tokens):
        if token in token_to_idx:
            encoded[i] = token_to_idx[token]
        else:
            encoded[i] = token_to_idx['<UNK>']
    return encoded


def encode_captions(data, token_to_idx, max_token_length):
    encoded_list = []
    for img in data:
        for phrase in img['description']:
            tokens = phrase
            if tokens is None: continue
            tokens_encoded = encode_caption(tokens, token_to_idx, max_token_length)
            encoded_list.append(tokens_encoded)
    return np.vstack(encoded_list)


def build_img_idx_to_phr_idxs(data):
    img_idx = 0
    phr_idx = 0
    num_images = len(data)
    img_to_first_phr = np.zeros(num_images, dtype=np.int32)
    img_to_last_phr = np.zeros(num_images, dtype=np.int32)
    for img in data:
        img_to_first_phr[img_idx] = phr_idx
        for phrase in img['description']:
            if phrase is None: continue
            phr_idx += 1
        img_to_last_phr[img_idx] = phr_idx - 1  # -1 to make these inclusive limits
        img_idx += 1

    return img_to_first_phr, img_to_last_phr


def build_filename_dict(data):
    # First make sure all filenames
    filenames_list = [img['filename'] for img in data]
    assert len(filenames_list) == len(set(filenames_list))

    next_idx = 0
    filename_to_idx, idx_to_filename = {}, {}
    for img in data:
        filename = img['filename']
        filename_to_idx[filename] = next_idx
        idx_to_filename[next_idx] = filename
        next_idx += 1
    return filename_to_idx, idx_to_filename


def add_images(data, h5_file, args):

    num_images = len(data)
    shape = (num_images, args.image_height, args.image_width, 3)
    image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)

    lock = Lock()
    q = Queue()

    for i, dic in enumerate(data):
        fname = dic['filename']
        filename = os.path.join(args.image_dir, fname)
        q.put((i, filename))

    def worker():
        while True:
            i, filename = q.get()
            img = imread(filename)
            # handle grayscale
            if img.ndim == 2:
                img = img[:, :, None][:, :, [0, 0, 0]]
            lock.acquire()
            if i % 1000 == 0:
                print('Writing image %d / %d' % (i, len(data)))
            image_dset[i] = img
            lock.release()
            q.task_done()

    print('adding images to hdf5.... (this might take a while)')
    for i in range(args.num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()


def words_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    replacements = {
        u'½': u'half',
        u'—': u'-',
        u'™': u'',
        u'¢': u'cent',
        u'ç': u'c',
        u'û': u'u',
        u'é': u'e',
        u'°': u' degree',
        u'…': u'',
    }
    punc_table = str.maketrans('è', 'e', string.punctuation)
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(punc_table).split()


def split_filter_captions(data, max_token_length, verbose=True):
    """
    Modifies data in-place by adding a 'tokens' field to each region.
    If the region's label is too long, 'tokens' will be None; otherwise
    it will be a list of strings.
    Splits by space when tokens_type = "words", or lists all chars when "chars"
    """
    captions_kept = 0
    captions_removed = 0
    lengths = []
    for i, img in enumerate(data):
        if verbose and (i + 1) % 2000 == 0:
            print('Splitting tokens in image %d / %d' % (i + 1, len(data)))
        regions_per_image = 0
        img_kept, img_removed = 0, 0
        for j, phrase in enumerate(img['description']):
            tokens = words_preprocess(phrase)
            # filter by length
            if (max_token_length > 0 and len(tokens) <= max_token_length) or max_token_length <= 0:
                data[i]['description'][j] = tokens
                captions_kept += 1
                img_kept += 1
                regions_per_image = regions_per_image + 1
                lengths.append(len(tokens))
            else:
                data[i]['description'][j] = None
                captions_removed += 1
                img_removed += 1
    if max_token_length <= 0:
        max_token_length = np.max(lengths)
    if verbose:
        print('Keeping %d captions' % captions_kept)
        print('Skipped %d captions for being too long' % captions_removed)
    return np.asarray(lengths, dtype=np.int32), max_token_length
def split_data(train_data, val_data, test_data):

        full_data = list(chain(*[train_data, val_data, test_data]))
        split = np.zeros(len(full_data), dtype=int)
        split[len(train_data):len(train_data)+len(val_data)] = 1
        split[len(train_data)+len(val_data):] = 2
        attr_csv = pd.read_csv(args.attr_data, index_col='image_id')
        idx_list = list(attr_csv.index)
        idx_array = list(idx_list.index(dic['filename']) for dic in full_data)
        attr_list = attr_csv.values[idx_array]
        attr_label = list(attr_csv.columns)
        return split, full_data, attr_list, attr_label

def main(args):
    # read in the data
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)
    with open(args.val_data, 'r') as g:
        val_data = json.load(g)
    with open(args.test_data, 'r') as h:
        test_data = json.load(h)
    # Only keep images that are in a split

    splits, data, attributes, attributes_labels = split_data(train_data, val_data, test_data)
    print('There are %d images total' % len(data))

    # create the output hdf5 file handle
    f = h5py.File(args.h5_output, 'w')
    f.create_dataset('split', data=splits)
    f.create_dataset('attributes', data=attributes)
    # add several fields to the file: images, and the original/resized widths/heights
    add_images(data, f, args)

    # process "label" field in each region to a "tokens" field, and cap at some max length
    lengths_vector, max_token_length = split_filter_captions(data, args.max_token_length)
    f.create_dataset('lengths', data=lengths_vector)
    # build vocabulary
    vocab = build_vocab(data, args.min_token_instances)  # vocab is a set()
    token_to_idx, idx_to_token = build_vocab_dict(vocab)  # both mappings are dicts

    # encode labels
    captions_matrix = encode_captions(data, token_to_idx, max_token_length)
    f.create_dataset('labels', data=captions_matrix)


    img_to_first_phr, img_to_last_phr = build_img_idx_to_phr_idxs(data)
    f.create_dataset('img_to_first_phr', data=img_to_first_phr)
    f.create_dataset('img_to_last_phr', data=img_to_last_phr)
    # integer mapping between image ids and box ids
    filename_to_idx, idx_to_filename = build_filename_dict(data)
    f.close()

    # and write the additional json file
    json_struct = {
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token,
        'filename_to_idx': filename_to_idx,
        'idx_to_filename': idx_to_filename,
        'attributes_labels': attributes_labels
    }
    with open(args.json_output, 'w') as f:
        json.dump(json_struct, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # INPUT settings
    parser.add_argument('--train_data',
                        default='data/my_clean_train_2.1.json',
                        help='Input JSON file with regions and captions')
    parser.add_argument('--val_data',
                        default='data/my_clean_dev_2.1.json',
                        help='Input JSON file with regions and captions')
    parser.add_argument('--test_data',
                        default='data/my_clean_test_2.1.json',
                        help='Input JSON file with regions and captions')
    parser.add_argument('--attr_data',
                        default='data/list_attr_celeba.csv',
                        help='Input csv file with images attributes')
    parser.add_argument('--image_dir',
                        default='data/img_align_celeba/img_align_celeba',
                        help='Directory containing all images')

    # OUTPUT settings
    parser.add_argument('--json_output',
                        default='data/face2text-dicts.json',
                        help='Path to output JSON file')
    parser.add_argument('--h5_output',
                        default='data/face2text-data.h5',
                        help='Path to output HDF5 file')

    # OPTIONS
    parser.add_argument('--image_height',
                        default=218, type=int,
                        help='Size of longest edge of preprocessed images')
    parser.add_argument('--image_width',
                        default=178, type=int,
                        help='Size of longest edge of preprocessed images')
    parser.add_argument('--max_token_length',
                        default=0, type=int,
                        help="Set to 0 to disable filtering")
    parser.add_argument('--min_token_instances',
                        default=1, type=int,
                        help="When token appears less than this times it will be mapped to <UNK>")
    parser.add_argument('--tokens_type', default='words',
                        help="Words|chars for word or char split in captions")
    parser.add_argument('--num_workers', default=1, type=int)
    args = parser.parse_args()
    main(args)

