import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

img = np.array(Image.open('AlexCap/data/img_align_celeba/img_align_celeba/000001.jpg'))
img = imgp[0].permute(1,2,0).cpu().detach().numpy()

# plt.show()
v=0
nb=19
boxes = data.gt_boxes[0].int().cpu().numpy()
for i ,boxe in enumerate(boxes[v:v+nb]):
    fig, ax = plt.subplots()

    ax.imshow(img)
    rect = patches.Rectangle((boxe[0]-(boxe[2]//2),boxe[1]-boxe[3]//2), boxe[2], boxe[3], edgecolor='b', facecolor='none')
    rect.set_label(captions[i+v])
    ax.add_patch(rect)

    plt.legend()
    plt.show()


np_gtboxes = data.gt_boxes[0].cpu().detach().numpy()
for boxe in np_gtboxes[v:v+nb]:
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((boxe[0]-(boxe[2]//2),boxe[1]-boxe[3]//2), boxe[2], boxe[3], edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

ori_npfeat = ori_roi_boxes.cpu().detach().numpy()
npfeat = roi_feat.cpu().detach().numpy()

plt.imshow(npfeat[0,1])
plt.show()
plt.imshow(ori_npfeat[0,0])
plt.show()