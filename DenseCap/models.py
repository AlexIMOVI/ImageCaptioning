import torch
from DenseCap.densecap.DenseCapModel import DenseCapModel  # Assuming DenseCapModel is implemented in another file
from DenseCap.densecap.RoiModel import RoiModel

class SetupModule:
    def __init__(self):
        pass

    @staticmethod
    def setup(opt):
        if opt['checkpoint_start_from'] == '':
            if opt.roi_only:
                print('Initializing a Roi model from scratch...')
                model = RoiModel(opt)
            else:
                print('Initializing a DenseCap model from scratch...')
                model = DenseCapModel(opt)
        else:
            print('Initializing a DenseCap model from', opt['checkpoint_start_from'])
            model = torch.load(opt['checkpoint_start_from'])['model']
            model.opt['end_objectness_weight'] = opt['end_objectness_weight']
            model.nets['localization_layer'].opt['mid_objectness_weight'] = opt['mid_objectness_weight']
            model.nets['localization_layer'].opt['mid_box_reg_weight'] = opt['mid_box_reg_weight']
            model.crits['box_reg_crit'].w = opt['end_box_reg_weight']
            rpn = model.nets['localization_layer'].nets['rpn']
            rpn.findModules('nn.RegularizeLayer')[0].w = opt['box_reg_decay']
            model.opt['train_remove_outbounds_boxes'] = opt['train_remove_outbounds_boxes']
            model.opt['captioning_weight'] = opt['captioning_weight']

            if torch.cuda.is_available():
                device = torch.device(model.device)
                torch.backends.cudnn.enabled = True
                model.net.to(device)
                model.nets['localization_layer'].nets['rpn'].to(device)
        # Find all Dropout layers and set their probabilities
        for i in range(len(model.nets['recog_base'])):
            if type(model.nets['recog_base'][i]) is torch.nn.modules.dropout.Dropout:
                model.nets['recog_base'][i].p = opt['drop_prob']

        model.float()

        return model

