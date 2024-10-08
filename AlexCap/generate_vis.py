from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as Fv
import torch.nn.functional as F
from math import ceil
import matplotlib.cm as cm
from AlexCap.eval.eval_resnet import score_captions
from easydict import EasyDict as edict
def generate_caption_vis(model, data, path, use_dataset_img):
    pred, alphas = model.forward_test(data)
    if use_dataset_img:
        gt_caption = model.llm.decode_sequence(data.gt_labels)[0]
        record = edict()
        record.candidate = pred[0]
        record.references = gt_caption
        blob = score_captions([record])
        meteor = blob['average_score']
        bleu = blob['average_bl_score']
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

    num_words = np.size(caption)
    w = np.round(np.sqrt(num_words)).astype(int)
    h = np.ceil(np.float32(num_words) / w).astype(int)
    alphas = alphas.squeeze(0)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    if use_dataset_img:
        caption_txt = f'GT: {gt_caption}'
        # caption_txt = f'GT: {gt_caption}\nPRED: {pred[0]}'
    else:
        caption_txt = f'PRED: {pred[0]}'
    text_obj = fig.text(0.5, 0.01, caption_txt, wrap=True, horizontalalignment='center', fontsize=12)
    renderer = fig.canvas.get_renderer()
    bbox = text_obj.get_window_extent(renderer=renderer)
    text_height = bbox.height / fig.dpi
    plt.subplots_adjust(bottom=text_height / 5.5 + 0.05)
    if use_dataset_img:
        plt.savefig(f'AlexCap/data/vis_results/{path[-10:-4]}_M{round(meteor*100,2)}_B{round(bleu*100,2)}.jpg')
    else:
        plt.savefig(f'AlexCap/data/vis_results/test.jpg')
    plt.show()
    ax1 = plt.subplot(w, h, 1)
    plt.imshow(img)
    plt.axis('off')
    for idx in range(num_words):
        ax2 = plt.subplot(w, h, idx + 1)
        label = caption[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=10)
        plt.text(0, 1, label, color='black', fontsize=10)
        plt.imshow(img)

        if 'ViTB' in model.opt.save_path:
            shape_size = 14
            scale = 16
        elif model.opt.use_vggface:
            shape_size = 14
            scale = 16
        else:
            shape_size = 7
            scale = 32
        alpha_img = F.interpolate(alphas[idx, :].reshape(1, 1, shape_size, shape_size), scale_factor=scale, mode='bilinear', align_corners=True)

        plt.imshow(alpha_img.squeeze().cpu(), alpha=0.8)
        plt.set_cmap(cm.get_cmap('Greys_r'))
        plt.axis('off')
    if use_dataset_img:
        plt.savefig(f'AlexCap/data/vis_results/{path[-10:-4]}_attention_M{round(meteor * 100, 2)}_B{round(bleu * 100, 2)}.jpg')
    plt.show()
