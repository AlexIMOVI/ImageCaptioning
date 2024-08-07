import torch
from DenseCap.densecap.DataLoader import DataLoader
from DenseCap.train_opts import get_config
from easydict import EasyDict as edict
from PIL import Image
import numpy as np
opt = get_config()
torch.set_default_dtype(torch.float32)
torch.manual_seed(opt['seed'])
if opt['gpu'] >= 0:
    device = opt['device']
    torch.cuda.manual_seed(opt['seed'])
    # torch.cuda.set_device(opt['gpu'])
torch.set_default_device(opt['device'])
loader = DataLoader(opt)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = Image.open('datasets/vg/VG_100K/53.jpg')
data = edict()
data.image, data.gt_boxes, data.gt_labels, info, data.region_proposals = loader.getBatch(opt, 1)

results = model.forward(img)
x = torch.from_numpy(np.array(img)).to(device).float().unsqueeze(0).permute(0,3,1,2)
for m in range(25):
    x = model._modules['model']._modules['model'].model[m](x)
    print(f'{m}: {x.size()}')