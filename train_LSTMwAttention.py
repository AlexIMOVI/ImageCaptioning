import torch
import time
import json
from easydict import EasyDict as edict
import torch.optim.adam as ad
from torch.optim.lr_scheduler import LambdaLR
from AlexCap.LSTMwAttentionModel import AlexCapModel
from AlexCap.AlexDataLoader import AlexDataLoader
from AlexCap.LSTMwAttention_opts import get_LSTMwAttention_config, name_LSTMwAttention_model
import AlexCap.eval.eval_resnet as eval_resnet
from AlexCap.generate_vis import generate_caption_vis
import numpy as np
torch.autograd.set_detect_anomaly(True)
torch.CUDA_LAUNCH_BLOCKING = 1
torch.set_default_dtype(torch.float32)

opt = get_LSTMwAttention_config()
torch.manual_seed(opt.seed)
if opt.gpu >= 0:
    device = opt.device
    torch.cuda.manual_seed(opt.seed)
torch.set_default_device(opt.device)
loader = AlexDataLoader(opt)
opt.seq_length = loader.getSeqLength()
opt.vocab_size = loader.getVocabSize()
opt.idx_to_token = loader.info['idx_to_token']
model = AlexCapModel(opt)
opt.loss_file, opt.result_file, opt.save_path = name_LSTMwAttention_model(opt)

if opt.from_checkpoint:
    print(f'loading trained model: {opt.save_path[30:-4]}\n')
    file = opt.save_path
    model.load_state_dict(torch.load(file))
    loss_history = json.load(open(opt.loss_file, 'r'))
    results_history = json.load(open(opt.result_file, 'r'))
    best_val_score = results_history[-1]['best_val_score']
    best_iter = results_history[-1]['best_iter']
    iter = best_iter
else:
    loss_history = []
    results_history = []
    iter = 0
    best_val_score = -1
    best_iter = 0
best_val_loss = 10
patience = 0
model.features.requires_grad_(False)
model.to(opt.device)
if opt.finetune_cnn and iter >= len(loader.train_ix) // opt.batch_size:
    if opt.use_vggface:
        model.features[10:].requires_grad_(True)
    else:
        model.features.requires_grad_(True)


def setup_scheduler(opt, model):
    params = model.parameters()
    optim = ad.Adam(params, opt.learning_rate, (opt.beta1, opt.beta2), opt.eps, weight_decay=opt.weight_decay)
    max_iter = (opt.save_checkpoint_every // opt.batch_size) * opt.num_epochs
    pad = opt.save_checkpoint_every // opt.batch_size**2
    warmup_steps = int(max_iter / opt.num_epochs)
    min_lr = opt.min_lr / opt.learning_rate
    if opt.use_scheduler:
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            else:
                cosine_decay = 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (max_iter - warmup_steps)))
                return max(min_lr, cosine_decay)
    else:
        def lr_lambda(current_step: int):
            return 1.0
    scheduler = LambdaLR(optim, lr_lambda)
    return optim, max_iter, pad, scheduler
def write_json(file, path):
    with open(path, 'w') as f:
        f.write('[')
        for i, item in enumerate(file):
            json.dump(item,f)
            if i != len(file)-1:
                f.write(',\n')
            else:
                f.write(']\n')
def lossFun():

    optim.zero_grad(set_to_none=True)
    model.train()
    data = edict()
    data.image, data.gt_labels, info, _ = loader.get_batch(opt, opt.batch_size)
    t1 = time.time()
    loss = model.forward_train(data)
    t2 = time.time()
    if opt.clip_grad:
        model.clip_gradient(1)
    losses_copy = {}
    losses_copy['captioning_loss'] = loss.item()
    losses_copy['epoch time in ms'] = (t2 - t1) * 1000
    if iter > 0 and iter % pad == 0:
        loss_history.append(losses_copy)
        write_json(loss_history, opt.loss_file)

    return losses_copy


optim, max_iter, pad, scheduler = setup_scheduler(opt, model)

while iter < max_iter:
    if opt.finetune_cnn and iter == len(loader.train_ix) // opt.batch_size:
        if opt.use_vggface:
            model.features[10:].requires_grad_(True)
        else:
            model.features.requires_grad_(True)

    losses = lossFun()
    optim.step()
    scheduler.step()
    x = ''
    for k, v in losses.items():
        x += f'{k}: {v:.5f}, '
    print(f'iter : {iter} -> {x}')
    if iter > 0 and ((iter + 1) % (opt.save_checkpoint_every // opt.batch_size) == 0 or iter + 1 == max_iter):
        eval_kwargs = {'model': model,
                       'loader': loader,
                       'split': 'val',
                       'max_images': -1,
                       'val_batch_size': 1}
        results = eval_resnet.eval_split(eval_kwargs)
        results_history.append(results)
        if results['ap_results']['meteor'] > best_val_score:
            torch.save(model.state_dict(), opt.save_path)
            best_val_score = results['ap_results']['meteor']
            best_iter = iter
        if results['loss_results'] < best_val_loss:
            best_val_loss = results['loss_results']
            patience = 0
        else:
            patience += 1
        results['best_val_score'] = best_val_score
        results['best_iter'] = best_iter
        write_json(results_history, opt.result_file)
        # print(f'patience = {patience}\n')
        # if patience > 10:
        #     break

    iter = iter + 1

"""MODEL EVALUATION ON TEST DATASET"""
eval_kwargs = {'model': model,
               'loader': loader,
               'split': 'test',
               'max_images': -1,
               'val_batch_size': 1}
results = eval_resnet.eval_split(eval_kwargs)

metlist = []
bleulist = []
model.llm.use_beam = True
for b in range(1, 6):
    model.llm.beam_size = b
    eval_kwargs = {'model': model,
                   'loader': loader,
                   'split': 'test',
                   'max_images': -1,
                   'val_batch_size': 1}
    results = eval_resnet.eval_split(eval_kwargs)
    metlist.append(results['ap_results']['meteor'])
    bleulist.append(results['ap_results']['bleu'])


loader_kwargs = {'split': 2, 'iterate': False}
data = edict()
data.image, data.gt_labels, info, _ = loader.get_batch(loader_kwargs, 1, args[4])
path = info[0]['filename'][0]
path = 'AlexCap/data/img_align_celeba/img_align_celeba/'+path
generate_caption_vis(model, data, path)

import numpy as np
import matplotlib.pyplot as plt
def display_logs(file, model_name, save=False):

    losses = [o['loss_results'] for o in file]
    step = file[0]['best_iter'] + 1
    steps = np.arange(step, len(file) * step + 1, step)
    meteor = [o['ap_results']['meteor'] for o in file]
    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].plot(steps, losses, 'bo-')
    ax[0].set_ylabel('loss')
    ax[0].set_title('Loss and METEOR score during training, on evaluation dataset')
    ax[1].plot(steps, meteor, 'go-')
    ax[1].set_ylabel('METEOR')
    fig.text(.5, .04, 'iter')
    if save:
        plt.savefig('AlexCap/graphs/'+model_name+'.png')
    plt.show()

name = opt.save_path[30:-4]
display_logs(results_history, name, True)

