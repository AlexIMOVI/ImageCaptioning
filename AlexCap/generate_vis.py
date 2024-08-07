from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as Fv
import torch.nn.functional as F
from math import ceil
import matplotlib.cm as cm

def generate_caption_vis(model, data, path, smooth=True):
    pred, alphas = model.forward_test(data)
    caption = pred[0].split()
    img = Image.open(path)
    w, h = img.size
    if w > h:
        w = w * 256 / h
        h = 256
    else:
        h = h * 256 / w
        w = 256
    left = (w - 224) // 2
    top = (h - 224) // 2
    resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
    img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
    img = img.astype('float32') / 255

    num_words = len(caption)
    w = np.round(np.sqrt(num_words)).astype(int)
    h = np.ceil(np.float32(num_words) / w).astype(int)
    alphas = alphas.squeeze(0)

    ax1 = plt.subplot(w, h, 1)
    plt.imshow(img)
    plt.axis('off')
    for idx in range(num_words):
        ax2 = plt.subplot(w, h, idx + 2)
        label = caption[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=10)
        plt.text(0, 1, label, color='black', fontsize=10)
        plt.imshow(img)

        if model.opt.use_vggface:
            shape_size = 14
            scale = 16
        else:
            shape_size = 7
            scale = 32
        if smooth:
            alpha_img = F.interpolate(alphas[idx, :].reshape(1, 1, shape_size, shape_size), scale_factor=scale, mode='bilinear', align_corners=True)
            # kernel_size = 4*20+1
            # alpha_img = Fv.gaussian_blur(alpha_img, kernel_size=[kernel_size, kernel_size], sigma=[20, 20])
        # else:
        #     alpha_img = skimage.transform.resize(alphas[idx, :].reshape(shape_size, shape_size),
        #                                          [img.shape[0], img.shape[1]])
        plt.imshow(alpha_img.squeeze().cpu(), alpha=0.8)
        plt.set_cmap(cm.get_cmap('Greys_r'))
        plt.axis('off')
    plt.show()