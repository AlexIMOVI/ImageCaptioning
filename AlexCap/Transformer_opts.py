from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict


def get_Transformer_config():
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
    cfg.embedding_size = 512
    cfg.Transformer_size = 512
    cfg.beta1 = 0.9
    cfg.beta2 = 0.999
    cfg.eps = 1e-8
    cfg.weight_decay = 0.1
    cfg.min_lr = 1e-6

    # Model checkpointing
    cfg.num_epochs = 50
    cfg.save_checkpoint_every = 8489
    cfg.save_path = "AlexCap/models_pth/best_model_Transformer.pth"
    cfg.loss_file = 'AlexCap/loss_logs/loss_history_Transformer.json'
    cfg.result_file = 'AlexCap/logs/results_history_Transformer.json'
    cfg.batch_size = 12
    cfg.clip_grad = True
    cfg.iterate = False
    cfg.from_checkpoint = False
    cfg.use_dropout = True
    cfg.drop_value = 0.1
    cfg.num_layers = 1
    cfg.finetune_cnn = True
    cfg.use_vggface = False
    # Misc
    cfg.id = ''
    cfg.seed = 123
    cfg.gpu = 0
    cfg.timing = False

    return cfg


def name_Transformer_model(opt):
    loss_name = opt.loss_file
    res_name = opt.result_file
    model_name = opt.save_path
    res_name = res_name.replace('Transformer', 'Transformer_clip') if opt.clip_grad else res_name
    res_name = res_name.replace('Transformer', 'Transformer_iter') if opt.iterate else res_name
    res_name = res_name.replace('Transformer', f'Transformer_bs{opt.batch_size}')
    res_name = res_name.replace('Transformer', f'Transformer_drop{opt.drop_value}') if opt.use_dropout else res_name
    res_name = res_name.replace('Transformer', 'Transformer_ft') if opt.finetune_cnn else res_name
    res_name = res_name.replace('Transformer', 'Transformer_vggface') if opt.use_vggface else res_name.replace('Transformer',
                                                                                                     'Transformer_resnet')
    model_name = model_name.replace('Transformer', 'Transformer_clip') if opt.clip_grad else model_name
    model_name = model_name.replace('Transformer', 'Transformer_iter') if opt.iterate else model_name
    model_name = model_name.replace('Transformer', f'Transformer_bs{opt.batch_size}')
    model_name = model_name.replace('Transformer', f'Transformer_drop{opt.drop_value}') if opt.use_dropout else model_name
    model_name = model_name.replace('Transformer', 'Transformer_ft') if opt.finetune_cnn else model_name
    model_name = model_name.replace('Transformer', 'Transformer_vggface') if opt.use_vggface else model_name.replace('Transformer',
                                                                                                           'Transformer_resnet')
    loss_name = loss_name.replace('Transformer', 'Transformer_clip') if opt.clip_grad else loss_name
    loss_name = loss_name.replace('Transformer', 'Transformer_iter') if opt.iterate else loss_name
    loss_name = loss_name.replace('Transformer', f'Transformer_bs{opt.batch_size}')
    loss_name = loss_name.replace('Transformer', f'Transformer_drop{opt.drop_value}') if opt.use_dropout else loss_name
    loss_name = loss_name.replace('Transformer', 'Transformer_ft') if opt.finetune_cnn else loss_name
    loss_name = loss_name.replace('Transformer', 'Transformer_vggface') if opt.use_vggface else loss_name.replace('Transformer',
                                                                                                        'Transformer_resnet')
    return loss_name, res_name, model_name

