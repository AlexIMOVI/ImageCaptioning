from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def get_config():
    cfg = edict()

    # Core ConvNet settings
    cfg.backend = 'cuda'
    cfg.device = 'cuda:0'
    # cfg.device = 'cpu'
    # Model settings
    cfg.rpn_hidden_dim = 512
    cfg.sampler_batch_size = 256
    cfg.rnn_size = 512
    cfg.input_encoding_size = 512
    cfg.sampler_high_thresh = 0.7
    cfg.sampler_low_thresh = 0.3
    cfg.train_remove_outbounds_boxes = 1

    # Loss function weights
    cfg.mid_box_reg_weight = 0.05
    cfg.mid_objectness_weight = 0.1
    cfg.end_box_reg_weight = 0.1
    cfg.end_objectness_weight = 0.1
    cfg.captioning_weight = 1.0
    cfg.weight_decay = 1e-6
    cfg.box_reg_decay = 5e-5

    # Data input settings
    cfg.data_h5 = 'data/VG-regions.h5'
    cfg.data_json = 'data/VG-regions-dicts.json'
    cfg.proposal_regions_h5 = ''
    cfg.debug_max_train_images = -1

    # Optimization
    cfg.learning_rate = 1e-5
    cfg.optim_beta1 = 0.9
    cfg.optim_beta2 = 0.999
    cfg.optim_epsilon = 1e-8
    cfg.drop_prob = 0.3
    cfg.max_iters = -1
    cfg.checkpoint_start_from = ''
    cfg.finetune_cnn_after = -1
    cfg.val_images_use = 10

    # Model checkpointing
    cfg.save_checkpoint_every = 20000
    cfg.save_path = "models_pth/best_model_transformer_gt.pth"
    cfg.loss_file = 'logs/loss_history_transformer_gt.json'
    cfg.result_file = 'logs/results_history_transformer_gt.json'
    cfg.from_checkpoint = True
    cfg.use_lstm = True
    cfg.num_layers = 1
    cfg.use_curriculum_learning = False
    cfg.use_dropout = False
    cfg.drop_value = 0.5
    cfg.finetune_cnn = True
    # Test-time model options
    cfg.test_rpn_nms_thresh = 0.7
    cfg.test_final_nms_thresh = 0.3
    cfg.test_num_proposals = 1000

    # Visualization
    cfg.progress_dump_every = 100
    cfg.losses_log_every = 10

    # Misc
    cfg.id = ''
    cfg.seed = 123
    cfg.gpu = 0
    cfg.timing = False
    cfg.clip_final_boxes = 1
    cfg.eval_first_iteration = 0

    return cfg

if __name__ == "__main__":
    config = get_config()
    print(config)

