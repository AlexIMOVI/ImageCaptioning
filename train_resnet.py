import torch
import json
import time
from AlexCap.my_train_opts import get_my_config
import torch.optim.adam as ad
from AlexCap.AlexDataLoader import AlexDataLoader
from easydict import EasyDict as edict
from AlexCap.ResnetCapModel import ResnetCapModel

torch.autograd.set_detect_anomaly(True)
torch.CUDA_LAUNCH_BLOCKING = 1
torch.set_default_dtype(torch.float32)
from_checkpoint = False

opt = get_my_config()
torch.manual_seed(opt['seed'])
if opt['gpu'] >= 0:
    device = opt['device']
    torch.cuda.manual_seed(opt['seed'])
torch.set_default_device(opt['device'])
loader = AlexDataLoader(opt)
opt.attributes_labels = loader.attributes_labels
model = ResnetCapModel(opt)
model.features.requires_grad_(False)
params = model.parameters()
optim = ad.Adam(params, opt['learning_rate'], weight_decay=opt['weight_decay'])


if from_checkpoint:
    file = opt.save_path
    model.load_state_dict(torch.load(file))
    loss_history = json.load(open(opt.loss_file, 'r'))
    results_history = json.load(open(opt.result_file, 'r'))
    best_val_score = results_history[-1]['best_val_score']
    iter = len(results_history) * opt.save_checkpoint_every
    best_iter = results_history[-1]['best_iter']
else:
    loss_history = []
    results_history = []
    iter = 0
    best_val_score = -1
    best_iter = 0
data = edict()

data.image, data.gt_labels, info, data.attr = loader.get_batch(opt)
iter = 0
max_iter = 10
pad = 50000
def lossFun():

    optim.zero_grad(set_to_none=True)
    model.train()
    data = edict()
    data.image, data.gt_labels, info, data.gt_attr = loader.get_batch(opt)
    t1 = time.time()
    loss = model.forward(data)
    t2 = time.time()
    losses_copy = {}
    losses_copy['captioning_loss'] = loss.item()
    # if iter > 0 and iter % pad == 0:
    #     loss_history.append(losses_copy)
    #     json.dump(loss_history, open(opt.loss_file, 'w'))
    losses_copy['epoch time in ms'] = (t2 - t1) * 1000
    return losses_copy

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as p:
    while iter < max_iter:
        losses = lossFun()
        optim.step()
        x = ''
        for k, v in losses.items():
                x += f'{k}: {v:.5f}, '
        print(f'iter : {iter}, -> {x}')
        iter = iter + 1
print(p.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
# file = "models_pth/best_model_attributes_pred.pth"
# # torch.save(model.state_dict(), file)
# model.load_state_dict(torch.load(file))