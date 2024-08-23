from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict


def get_LSTMwAttention_config():
    cfg = edict()

    # Core ConvNet settings
    cfg.backend = 'cuda'
    cfg.device = 'cuda:0'


    # Data input settings
    cfg.data_h5 = 'data/face2text-data.h5'
    cfg.data_json = 'data/face2text-dicts.json'
    cfg.debug_max_train_images = -1


    # Optimization
    cfg.use_scheduler = True
    cfg.learning_rate = 3e-4
    cfg.embedding_size = 1024
    cfg.lstm_size = 768
    cfg.beta1 = 0.9
    cfg.beta2 = 0.999
    cfg.eps = 1e-8
    cfg.weight_decay = 1e-6
    cfg.min_lr = 1e-6

    # Model checkpointing
    cfg.num_epochs = 50
    cfg.save_checkpoint_every = 8489
    cfg.save_path = "AlexCap/models_pth/best_model_sch_LSTMwAttention.pth"
    cfg.loss_file = 'AlexCap/loss_logs/loss_history_sch_LSTMwAttention.json'
    cfg.result_file = 'AlexCap/logs/results_history_sch_LSTMwAttention.json'
    cfg.batch_size = 12
    cfg.clip_grad = True
    cfg.iterate = False
    cfg.from_checkpoint = False
    cfg.use_dropout = False
    cfg.drop_value = 0.5
    cfg.finetune_cnn = True
    cfg.use_vggface = True
    # Misc
    cfg.id = ''
    cfg.seed = 123
    cfg.gpu = 0
    cfg.timing = False

    return cfg


def name_LSTMwAttention_model(opt):
    loss_name = opt.loss_file
    res_name = opt.result_file
    model_name = opt.save_path
    res_name = res_name.replace('LSTMwAttention', 'LSTMwAttention_clip') if opt.clip_grad else res_name
    res_name = res_name.replace('LSTMwAttention', 'LSTMwAttention_iter') if opt.iterate else res_name
    res_name = res_name.replace('LSTMwAttention', f'LSTMwAttention_bs{opt.batch_size}')
    res_name = res_name.replace('LSTMwAttention', f'LSTMwAttention_drop{opt.drop_value}') if opt.use_dropout else res_name
    res_name = res_name.replace('LSTMwAttention', 'LSTMwAttention_ft') if opt.finetune_cnn else res_name
    res_name = res_name.replace('LSTMwAttention', 'LSTMwAttention_vggface') if opt.use_vggface else res_name.replace('LSTMwAttention',
                                                                                                     'LSTMwAttention_resnet')
    model_name = model_name.replace('LSTMwAttention', 'LSTMwAttention_clip') if opt.clip_grad else model_name
    model_name = model_name.replace('LSTMwAttention', 'LSTMwAttention_iter') if opt.iterate else model_name
    model_name = model_name.replace('LSTMwAttention', f'LSTMwAttention_bs{opt.batch_size}')
    model_name = model_name.replace('LSTMwAttention', f'LSTMwAttention_drop{opt.drop_value}') if opt.use_dropout else model_name
    model_name = model_name.replace('LSTMwAttention', 'LSTMwAttention_ft') if opt.finetune_cnn else model_name
    model_name = model_name.replace('LSTMwAttention', 'LSTMwAttention_vggface') if opt.use_vggface else model_name.replace('LSTMwAttention',
                                                                                                           'LSTMwAttention_resnet')
    loss_name = loss_name.replace('LSTMwAttention', 'LSTMwAttention_clip') if opt.clip_grad else loss_name
    loss_name = loss_name.replace('LSTMwAttention', 'LSTMwAttention_iter') if opt.iterate else loss_name
    loss_name = loss_name.replace('LSTMwAttention', f'LSTMwAttention_bs{opt.batch_size}')
    loss_name = loss_name.replace('LSTMwAttention', f'LSTMwAttention_drop{opt.drop_value}') if opt.use_dropout else loss_name
    loss_name = loss_name.replace('LSTMwAttention', 'LSTMwAttention_ft') if opt.finetune_cnn else loss_name
    loss_name = loss_name.replace('LSTMwAttention', 'LSTMwAttention_vggface') if opt.use_vggface else loss_name.replace('LSTMwAttention',
                                                                                                        'LSTMwAttention_resnet')
    return loss_name, res_name, model_name

