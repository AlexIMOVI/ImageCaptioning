from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict


def get_my_config():
    cfg = edict()

    # Core ConvNet settings
    cfg.backend = 'cuda'
    cfg.device = 'cuda:0'

    cfg.weight_decay = 1e-6
    # Data input settings
    cfg.data_h5 = 'data/face2text-data.h5'
    cfg.data_json = 'data/face2text-dicts.json'
    cfg.debug_max_train_images = -1

    # Optimization
    cfg.learning_rate = 1e-5
    cfg.embedding_size = 512
    cfg.lstm_size = 512
    cfg.beta1 = 0.9
    cfg.beta2 = 0.999
    cfg.eps = 1e-8

    # Model checkpointing
    cfg.save_checkpoint_every = 8489
    cfg.save_path = "AlexCap/models_pth/best_model_celeba.pth"
    cfg.loss_file = 'AlexCap/loss_logs/loss_history_celeba.json'
    cfg.result_file = 'AlexCap/logs/results_history_celeba.json'
    cfg.batch_size = 12
    cfg.iterate = False
    cfg.from_checkpoint = True
    cfg.use_lstm = True
    cfg.num_layers = 1
    cfg.use_curriculum_learning = False
    cfg.use_dropout = False
    cfg.drop_value = 0.2
    cfg.finetune_cnn = True
    cfg.use_vggface = True
    cfg.clip_grad = False
    cfg.use_attention = False
    # Misc
    cfg.id = ''
    cfg.seed = 123
    cfg.gpu = 0
    cfg.timing = False

    return cfg


def name_model(opt):
    loss_name = opt.loss_file
    res_name = opt.result_file
    model_name = opt.save_path
    res_name = res_name.replace('celeba', 'celeba_clip') if opt.clip_grad else res_name
    res_name = res_name.replace('celeba', 'celeba_rand') if not opt.iterate else res_name
    res_name = res_name.replace('celeba', 'celeba_cvlearn') if opt.use_curriculum_learning else res_name
    res_name = res_name.replace('celeba', f'celeba_bs{opt.batch_size}')
    res_name = res_name.replace('celeba', f'celeba_drop{opt.drop_value}') if opt.use_dropout else res_name
    res_name = res_name.replace('celeba', 'celeba_ft') if opt.finetune_cnn else res_name
    res_name = res_name.replace('celeba', 'celeba_attention') if opt.use_attention else res_name
    res_name = res_name.replace('celeba', 'celeba_lstm') if opt.use_lstm else res_name.replace('celeba',
                                                                                             'celeba_transformer')
    res_name = res_name.replace('celeba', 'celeba_vggface') if opt.use_vggface else res_name.replace('celeba',
                                                                                                   'celeba_resnet')
    model_name = model_name.replace('celeba', 'celeba_clip') if opt.clip_grad else model_name
    model_name = model_name.replace('celeba', 'celeba_rand') if not opt.iterate else model_name
    model_name = model_name.replace('celeba', 'celeba_cvlearn') if opt.use_curriculum_learning else model_name
    model_name = model_name.replace('celeba', f'celeba_bs{opt.batch_size}')
    model_name = model_name.replace('celeba', f'celeba_drop{opt.drop_value}') if opt.use_dropout else model_name
    model_name = model_name.replace('celeba', 'celeba_ft') if opt.finetune_cnn else model_name
    model_name = model_name.replace('celeba', 'celeba_attention') if opt.use_attention else model_name
    model_name = model_name.replace('celeba', 'celeba_lstm') if opt.use_lstm else model_name.replace('celeba',
                                                                                                   'celeba_transformer')
    model_name = model_name.replace('celeba', 'celeba_vggface') if opt.use_vggface else model_name.replace('celeba',
                                                                                                         'celeba_resnet')
    loss_name = loss_name.replace('celeba', 'celeba_clip') if opt.clip_grad else loss_name
    loss_name = loss_name.replace('celeba', 'celeba_rand') if not opt.iterate else loss_name
    loss_name = loss_name.replace('celeba', 'celeba_cvlearn') if opt.use_curriculum_learning else loss_name
    loss_name = loss_name.replace('celeba', f'celeba_bs{opt.batch_size}')
    loss_name = loss_name.replace('celeba', f'celeba_drop{opt.drop_value}') if opt.use_dropout else loss_name
    loss_name = loss_name.replace('celeba', 'celeba_ft') if opt.finetune_cnn else loss_name
    loss_name = loss_name.replace('celeba', 'celeba_attention') if opt.use_attention else loss_name
    loss_name = loss_name.replace('celeba', 'celeba_lstm') if opt.use_lstm else loss_name.replace('celeba',
                                                                                                'celeba_transformer')
    loss_name = loss_name.replace('celeba', 'celeba_vggface') if opt.use_vggface else loss_name.replace('celeba',
                                                                                                      'celeba_resnet')
    return loss_name, res_name, model_name


if __name__ == "__main__":
    config = get_my_config()
    print(config)
