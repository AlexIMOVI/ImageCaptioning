import torch
import time
import json
from AlexGTModel.AlexDenseModel import AlexCapModel
from easydict import EasyDict as edict
import torch.optim.adam as ad
from AlexGTModel.DataLoader import DataLoader
from AlexGTModel.train_opts import get_config
import AlexGTModel.eval.eval_gt as eval_gt

torch.autograd.set_detect_anomaly(True)
torch.CUDA_LAUNCH_BLOCKING = 1
torch.set_default_dtype(torch.float32)

opt = get_config()
torch.manual_seed(opt['seed'])
if opt['gpu'] >= 0:
    device = opt['device']
    torch.cuda.manual_seed(opt['seed'])
torch.set_default_device(opt['device'])
loader = DataLoader(opt)
opt.seq_length = loader.getSeqLength()
opt.vocab_size = loader.getVocabSize()
opt.idx_to_token = loader.info['idx_to_token']
model = AlexCapModel(opt)
if opt.use_lstm:
    opt.loss_file = opt.loss_file.replace("transformer", "lstm")
    opt.result_file = opt.result_file.replace("transformer", "lstm")
    opt.save_path = opt.save_path.replace("transformer", "lstm")
if opt.use_dropout:
    opt.loss_file = opt.loss_file.replace("gt", f'gt_drop{opt.drop_value}')
    opt.result_file = opt.result_file.replace("gt", f'gt_drop{opt.drop_value}')
    opt.save_path = opt.save_path.replace("gt", f'gt_drop{opt.drop_value}')
if opt.finetune_cnn:
    opt.loss_file = opt.loss_file.replace("gt", 'gt_finetuned')
    opt.result_file = opt.result_file.replace("gt", 'gt_finetuned')
    opt.save_path = opt.save_path.replace("gt", 'gt_finetuned')

max_iter = 800000
pad = 500

if opt.from_checkpoint:
    file = opt.save_path
    model.load_state_dict(torch.load(file))
    loss_history = json.load(open(opt.loss_file, 'r'))
    results_history = json.load(open(opt.result_file, 'r'))
    best_val_score = results_history[-1]['best_val_score']
    iter = len(results_history) * opt.save_checkpoint_every
    best_iter = results_history[-1]['best_iter']
    loss_history = loss_history[:iter//pad]
    loader.iterators[0] = iter % len(loader.train_ix)
else:
    loss_history = []
    results_history = []
    iter = 0
    best_val_score = -1
    best_iter = 0

model.features.requires_grad_(False)
model.to(opt['device'])
params = model.parameters()
optim = ad.Adam(params, opt['learning_rate'], weight_decay=opt['weight_decay'])
if opt.finetune_cnn and iter >= len(loader.train_ix):
    model.features[10:].requires_grad_(True)

def lossFun():

    optim.zero_grad(set_to_none=True)
    model.train()
    data = edict()
    data.image, data.gt_boxes, data.gt_labels, info = loader.get_batch(opt)
    if opt.use_curriculum_learning:
        model.llm.teacher_prob = 40000/(40000+torch.exp(iter/40000))
    t1 = time.time()
    loss = model.forward_train(data)
    t2 = time.time()
    losses_copy = {}
    losses_copy['captioning_loss'] = loss.item()
    if iter > 0 and iter % pad == 0:
        loss_history.append(losses_copy)
        json.dump(loss_history, open(opt.loss_file, 'w'))
    losses_copy['epoch time in ms'] = (t2 - t1) * 1000
    return losses_copy

while iter < max_iter:
    if opt.finetune_cnn and iter == len(loader.train_ix):
        model.features[10:].requires_grad_(True)

    losses = lossFun()
    optim.step()
    x = ''
    for k, v in losses.items():
        x += f'{k}: {v:.5f}, '
    print(f'iter : {iter}, -> {x}')
    if iter > 0 and ((iter + 1) % opt.save_checkpoint_every == 0 or iter + 1 == max_iter):
        eval_kwargs = {'model': model,
                       'loader': loader,
                       'split': 'val',
                       'max_images': -1}

        results = eval_gt.eval_split(eval_kwargs)
        results_history.append(results)
        if results['ap_results']['map'] > best_val_score:
            torch.save(model.state_dict(), opt.save_path)
            best_val_score = results['ap_results']['map']
            best_iter = iter
        results['best_val_score'] = best_val_score
        results['best_iter'] = best_iter
        json.dump(results_history, open(opt.result_file, 'w'))
    iter = iter + 1

"""MODEL EVALUATION ON TEST DATASET"""

# eval_kwargs = {'model': model,
#                'loader': loader,
#                'split': 'test',
#                'max_images': -1}
# results = eval_gt.eval_split(eval_kwargs)
