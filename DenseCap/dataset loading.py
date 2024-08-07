from datasets import load_dataset
import numpy as np
import torch
import string
dataset = load_dataset("bookcorpus")
w_list = np.array(['text'],dtype=str)

for j in range(100000):
    w_list = np.r_[w_list,dataset['train'][2]['text'].lower().translate(string.punctuation).split()]
    w_list = np.unique(w_list)

w_dict = {}
for i in range(len(w_list)):
    w_dict[w_list[i]] = i

# with open("dict_table.json", "w") as outfile:
#     json.dump(w_table, outfile)