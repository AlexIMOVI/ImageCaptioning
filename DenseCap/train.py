import torch
import time
import json
from DenseCap.densecap.DataLoader import DataLoader
from DenseCap.densecap.densecap_utils import utils
from train_opts import get_config
from models import SetupModule
import DenseCap.eval.eval_utils as eval_utils
from easydict import EasyDict as edict
import torch.optim.adam as ad

opt = get_config()
torch.set_default_dtype(torch.float32)
torch.manual_seed(opt['seed'])
if opt['gpu'] >= 0:
    device = opt['device']
    torch.cuda.manual_seed(opt['seed'])
    # torch.cuda.set_device(opt['gpu'])
torch.set_default_device(opt['device'])
loader = DataLoader(opt)
opt.seq_length = loader.getSeqLength()
opt.vocab_size = loader.getVocabSize()
opt.idx_to_token = loader.info['idx_to_token']

opt.use_transformer = True
opt.roi_only = False
opt.use_resnet = False
torch.CUDA_LAUNCH_BLOCKING = 1
dtype = 'torch.CudaTensor'
model = SetupModule.setup(opt).to(device)
loss_file = 'loss_history_anchors_detach.json'
result_file = 'results_history_anchors_detach.json'
torch.autograd.set_detect_anomaly(True)
params = model.parameters()
optim = ad.Adam(params, opt['learning_rate'], weight_decay=opt['weight_decay'])
iter = 0
loss_history = []
results_history = []
best_val_score = -1
max_iter = 200000
pad = max_iter//1000
do_eval = True
best_iter = 0
def lossFun():

    optim.zero_grad(set_to_none=True)
    model.train()
    data = edict()
    data.image, data.gt_boxes, data.gt_labels, info, data.region_proposals = loader.get_batch(opt)
    t1 = time.time()
    losses = model.forward_backward(data)
    t2 = time.time()
    if opt['losses_log_every'] > 0 and iter % pad == 0:
        losses_copy = {}
        for k, v in losses.items():
            losses_copy[k] = v.item()
        loss_history.append(losses_copy)
        json.dump(loss_history, open(loss_file, 'w'))
    losses['epoch time in ms'] = (t2 - t1) * 1000
    return losses

model.net.conv_net1.requires_grad_(False)
model.net.conv_net2.requires_grad_(False)
model.nets['recog_base'].requires_grad_(False)
# stop_event = threading.Event()
# def key_listener():
#     input("Press space and Enter to stop training...\n")
#     stop_event.set()
#
#
# stop_event.clear()
# listener = threading.Thread(target=key_listener)
# listener.start()
while iter < max_iter:
    losses = lossFun()
    optim.step()

    print(f'iter : {iter}, -> {utils.build_loss_string(losses)}')
    if do_eval and iter > 0 and (iter % opt['save_checkpoint_every'] == 0 or iter + 1 == opt['max_iters']):
        eval_kwargs = {'model': model,
                       'loader': loader,
                       'split': 'val',
                       'max_images': 1000}
        results = eval_utils.eval_split(eval_kwargs)
        results_history.append(results)

        if results['ap_results']['map'] > best_val_score:
            save_path = "best_model_transformer_no_detach.pth"
            torch.save(model.state_dict(), save_path)
            best_val_score = results['ap_results']['map']
            best_iter = iter
        results['best_val_score'] = best_val_score
        results['best_iter'] = best_iter
        json.dump(results_history, open(result_file, 'w'))
    iter = iter + 1
#     if stop_event.is_set():
#         break
# listener.join()



"""EVAL PART"""



"""SAVE MODEL BACKWARD GRAPH"""
# make_dot(loss_history[-1]['total_loss'], params = dict(model.named_parameters())).render("networks/transfo_total", format="jpg")


"""LOSS DISPLAY"""

# for k in loss_history[0].keys():
#     display_loss_history(loss_history, k, f'transformer_best_model_{best_iter}', pad)

"""SAVING AND LOADING MODEL"""

# file = "best_model_transformer_anchors_detach.pth"
# # torch.save(model.state_dict(), file)
# model.load_state_dict(torch.load(file))
# eval_kwargs = {'model': model,
#                'loader': loader,
#                'split': 'test',
#                'max_images': 10,
#                'dtype': dtype}
# results = eval_utils.eval_split(eval_kwargs)

"""PREDICTION TEST"""
# model.eval()
# data = edict()
# opt['split'] = 2
# data.image, data.gt_boxes, data.gt_labels, info, data.region_proposals = loader.get_batch(opt,2)
# boxes, logprobs, captions = model.forward_test(data.image)

# evaluator = eval_utils.DenseCaptioningEvaluator(opt)
# evaluator.addResult(logprobs, boxes, captions, data.gt_boxes.squeeze(), model.nets['llm'].decode_sequence(data.gt_labels.squeeze()))
# preds = model.out[4]
# gt = model.out[6]
# gt = data.gt_labels.squeeze()
# # rec = [[],[]]
# rec = []
# for i in range(gt.size(0)):
#     # p1 = torch.nn.LogSoftmax(dim=1)(preds[i])
#     gt_seq = gt[i].unsqueeze(0)
#     # pv = torch.max(p1,1).indices.unsqueeze(0)
#     # cap = model.nets['llm'].decode_sequence(pv)
#     cap_gt = model.nets['llm'].decode_sequence(gt_seq)
#     # rec[0].append(cap)
#     rec.append(cap_gt)
#     # print(f'gt = {cap_gt}')

