from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict


def get_paper_config():
    cfg = edict()

    # Core ConvNet settings
    cfg.backend = 'cuda'
    cfg.device = 'cuda:0'


    # Data input settings
    cfg.data_h5 = 'data/face2text-data.h5'
    cfg.data_json = 'data/face2text-dicts.json'
    cfg.debug_max_train_images = -1


    # Optimization
    cfg.learning_rate = 1e-4
    cfg.embedding_size = 512
    cfg.lstm_size = 768
    cfg.beta1 = 0.9
    cfg.beta2 = 0.98
    cfg.eps = 1e-9
    cfg.weight_decay = 1e-6

    # Model checkpointing
    cfg.save_checkpoint_every = 8489
    cfg.save_path = "AlexCap/models_pth/best_model_paper.pth"
    cfg.loss_file = 'AlexCap/loss_logs/loss_history_paper.json'
    cfg.result_file = 'AlexCap/logs/results_history_paper.json'
    cfg.batch_size = 12
    cfg.clip_grad = True
    cfg.iterate = False
    cfg.from_checkpoint = False
    cfg.use_lstm = False
    cfg.num_layers = 1
    cfg.use_curriculum_learning = False
    cfg.use_dropout = False
    cfg.drop_value = 0.5
    cfg.finetune_cnn = True
    cfg.use_vggface = False
    cfg.use_attention = False
    # Misc
    cfg.id = ''
    cfg.seed = 123
    cfg.gpu = 0
    cfg.timing = False

    return cfg


def name_paper_model(opt):
    loss_name = opt.loss_file
    res_name = opt.result_file
    model_name = opt.save_path
    res_name = res_name.replace('paper', 'paper_clip') if opt.clip_grad else res_name
    res_name = res_name.replace('paper', 'paper_rand') if not opt.iterate else res_name
    res_name = res_name.replace('paper', 'paper_cvlearn') if opt.use_curriculum_learning else res_name
    res_name = res_name.replace('paper', f'paper_bs{opt.batch_size}')
    res_name = res_name.replace('paper', f'paper_drop{opt.drop_value}') if opt.use_dropout else res_name
    res_name = res_name.replace('paper', 'paper_ft') if opt.finetune_cnn else res_name
    res_name = res_name.replace('paper', 'paper_attention') if opt.use_attention else res_name
    res_name = res_name.replace('paper', 'paper_lstm') if opt.use_lstm else res_name.replace('paper',
                                                                                               'paper_transformer')
    res_name = res_name.replace('paper', 'paper_vggface') if opt.use_vggface else res_name.replace('paper',
                                                                                                     'paper_resnet')
    model_name = model_name.replace('paper', 'paper_clip') if opt.clip_grad else model_name
    model_name = model_name.replace('paper', 'paper_rand') if not opt.iterate else model_name
    model_name = model_name.replace('paper', 'paper_cvlearn') if opt.use_curriculum_learning else model_name
    model_name = model_name.replace('paper', f'paper_bs{opt.batch_size}')
    model_name = model_name.replace('paper', f'paper_drop{opt.drop_value}') if opt.use_dropout else model_name
    model_name = model_name.replace('paper', 'paper_ft') if opt.finetune_cnn else model_name
    model_name = model_name.replace('paper', 'paper_attention') if opt.use_attention else model_name
    model_name = model_name.replace('paper', 'paper_lstm') if opt.use_lstm else model_name.replace('paper',
                                                                                                     'paper_transformer')
    model_name = model_name.replace('paper', 'paper_vggface') if opt.use_vggface else model_name.replace('paper',
                                                                                                           'paper_resnet')
    loss_name = loss_name.replace('paper', 'paper_clip') if opt.clip_grad else loss_name
    loss_name = loss_name.replace('paper', 'paper_rand') if not opt.iterate else loss_name
    loss_name = loss_name.replace('paper', 'paper_cvlearn') if opt.use_curriculum_learning else loss_name
    loss_name = loss_name.replace('paper', f'paper_bs{opt.batch_size}')
    loss_name = loss_name.replace('paper', f'paper_drop{opt.drop_value}') if opt.use_dropout else loss_name
    loss_name = loss_name.replace('paper', 'paper_ft') if opt.finetune_cnn else loss_name
    loss_name = loss_name.replace('paper', 'paper_attention') if opt.use_attention else loss_name
    loss_name = loss_name.replace('paper', 'paper_lstm') if opt.use_lstm else loss_name.replace('paper',
                                                                                                     'paper_transformer')
    loss_name = loss_name.replace('paper', 'paper_vggface') if opt.use_vggface else loss_name.replace('paper',
                                                                                                        'paper_resnet')
    return loss_name, res_name, model_name

