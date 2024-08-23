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
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from io import StringIO
import matplotlib.pyplot as plt


def change_extension(file_path, new_extension):
    base = os.path.splitext(file_path)[0]
    return base + new_extension

def save_as_np(file_list, temp_dir):
    np_file_list = []
    for file in file_list:
        try:
            print(f"Chargement du fichier : {file}")
            with open(file, 'r', encoding='utf-8') as f:
                contenu = f.read()
                contenu = contenu.replace(',', '.')
                df = pd.read_csv(StringIO(contenu), delimiter=r'\s+', header=None)
                matrice = df.to_numpy(dtype=float)
                print(f"Matrice depuis {file} : {matrice}")
                file = change_extension(file, '.npy')
                np_file = os.path.join(temp_dir, os.path.basename(file))
                print(np_file)
                np_file_list.append(np_file)
                np.save(np_file, matrice)
        except Exception as e:
            print(f"Erreur lors du chargement du fichier {file} : {e}")
            return None
    return np_file_list

data_frame = pd.read_csv('finale_test.csv', sep=';')
temp_dir = tempfile.mkdtemp()

file_list = data_frame['luminance'].tolist()
labels = data_frame['Intensity'].tolist()
print('afficher labels',labels)
np_file_list = save_as_np(file_list, temp_dir)

def load_npy_file(file_path, label):
    data = np.load(file_path)
    #data = data * 60
    data = data.astype(np.float32)  # Assurez-vous que les données sont en float32
    return data, label

def load_data(file_path, label):
    data, label = tf.numpy_function(load_npy_file, [file_path, label], [tf.float32, tf.float32])
    return data, label

# Créer le dataset
dataset = tf.data.Dataset.from_tensor_slices((np_file_list, labels))
dataset = dataset.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
shutil.rmtree(temp_dir)
# Convertir le dataset en tableaux NumPy
data_list = []
label_list = []
for data, label in dataset:
    data_list.append(data.numpy())
    label_list.append(label.numpy())

data_array = np.array(data_list)
print('afficher data_array',data_array)
label_array = np.array(label_list)
print('afficher label_array',label_array)


# Diviser les données
X_train, X_val, y_train, y_val = train_test_split(data_array, label_array, test_size=0.2, random_state=42)

print('Forme de X_train:', X_train.shape)
print('Forme de X_val:', X_val.shape)
print('Forme de y_train:', y_train.shape)
print('Forme de y_val:', y_val.shape)

# Redimensionner les données pour l'entrée du CNN
X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], 1)

X_train = X_train.astype(np.float16)
X_val = X_val.astype(np.float16)

# Créer le modèle
# Définir la forme d'entrée
input_shape = (X_train.shape[1], X_train.shape[2], 1)

# Créer le modèle en utilisant un objet Input
model = Sequential([
    Input(shape=input_shape),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.summary()

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=["accuracy"])
'''
# Créer un dataset à partir des données
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(20).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(20).prefetch(tf.data.AUTOTUNE)
'''
# Utiliser le dataset dans model.fit
history = model.fit(X_train,y_train, epochs=10, batch_size=10,validation_data=(X_val,y_val))

# Évaluer le modèle
mse = model.evaluate(X_val, y_val)
print("Erreur Quadratique Moyenne sur l'ensemble de test :", mse)
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
#start = 2  # pour ne pas afficher les valeurs aux premières epochs
epochs = range (1,11)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()