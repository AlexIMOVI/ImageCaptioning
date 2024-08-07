import torch
import torch.nn as nn
from DenseCap.densecap.BilinearRoiPooling import BilinearRoiPooling
from DenseCap.densecap.DataLoader import DataLoader
from train_opts import get_config
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from DenseCap.densecap.BoxToAffine import BoxToAffine
from DenseCap.densecap.BatchBilinearSamplerBHWD import BatchBilinearSamplerBHWD,AffineGridGeneratorBHWD
from DenseCap.densecap.net_utils import load_cnn, subsequence
opt = get_config()
torch.set_default_dtype(torch.float32)
torch.manual_seed(opt['seed'])
if opt['gpu'] >= 0:
    device = opt['device']
    torch.cuda.manual_seed(opt['seed'])
    # torch.cuda.set_device(opt['gpu'])
torch.set_default_device(opt['device'])
loader = DataLoader(opt)
data = edict()
data.image, data.gt_boxes, data.gt_labels, info, data.region_proposals = loader.getBatch(opt, 1)
H, W = data.image.size(2), data.image.size(3)
HH, WW = 128, 128


class IdentityConv(nn.Module):
    def __init__(self, C, k=3):
        super(IdentityConv, self).__init__()
        assert k % 2 == 1, "Kernel size must be odd"
        pad = k // 2
        self.conv = nn.Conv2d(C, C, kernel_size=k, stride=1, padding=pad)
        # Initialize the weights and biases
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        center = k // 2
        for i in range(C):
            self.conv.weight.data[i, i, center, center] = 1

    def forward(self, x):
        return self.conv(x)


net = nn.Sequential(
    IdentityConv(3, 3),
    nn.AvgPool2d(kernel_size=2, stride=2),
    IdentityConv(3, 3),
    nn.AvgPool2d(kernel_size=2, stride=2),
    IdentityConv(3, 3),
    nn.AvgPool2d(kernel_size=2, stride=2)
)

img_small = net(data.image)
# img_small = img_small.squeeze(0).permute(1, 2, 0).int().cpu().detach().numpy()
# plt.imshow(img_small)
# plt.show()
cnn = subsequence(load_cnn('vgg-16'), 0, 30)
# boxes = torch.tensor([
#     [(W + 1) / 2, (H + 1) / 2, W, H],
#     [(W + 1) / 2, (H + 1) / 2, W / 2, H / 2],
#     [(W + 1) / 2 - 100, (H + 1) / 2, W / 2, H / 2],
#     [(W + 1) / 2, (H + 1) / 2, W, H / 2],
#     [(W + 1) / 2, (H + 1) / 2 - 100, W / 2, H / 2],
#     [(W + 1) / 2, (H + 1) / 2, W / 2, H]
# ])
boxes = torch.tensor([
    [(3*W - 1) / 4, (H - 1) / 2, W / 2, H]])
boxtoaff = BoxToAffine()
boxtoaff.setSize(H, W)
params = boxtoaff.forward(boxes)
grids = AffineGridGeneratorBHWD(HH, WW, opt['device']).forward(params)
im = cnn.forward(data.image)
bhwd = BatchBilinearSamplerBHWD()
samples1 = bhwd.forward([data.image[0], grids])
samples2 = bhwd.forward([im[0], grids])

for t in samples1:
    img = t.permute(1, 2, 0).int().cpu().detach().numpy()[:, :, ::-1]
    plt.imshow(img)
    plt.show()

for t in samples2:
    img = t[0:1].permute(1, 2, 0).int().cpu().detach().numpy()[:, :, ::-1]
    plt.imshow(img)
    plt.show()
plt.imshow(im[0,0:1].permute(1, 2, 0).int().cpu().detach().numpy()[:, :, ::-1])
plt.show()
roi = BilinearRoiPooling(HH, WW, opt['device'])
roi.setImageSize(H, W)
samples1 = roi.forward([data.image[0], boxes])
roi.setImageSize(H, W)
samples2 = roi.forward([img_small[0], boxes])
