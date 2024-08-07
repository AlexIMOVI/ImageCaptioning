import torchvision
import torch
import torch.nn as nn
from LanguageModel import LanguageModel
from Transformer import Transformer
from LocalizationLayer import LocalizationLayer
from BoxRegressionCriterion import BoxRegressionCriterion
from ApplyBoxTransform import ApplyBoxTransform
from LogisticCriterion import LogisticCriterion
from PosSlicer import PosSlicer
import box_utils
from densecap_utils import utils
import net_utils
from LSTMLoss import CustomCrossEntropyLoss
class RecogNet(nn.Module):
    def __init__(self, opt, kwargs):
        super(RecogNet, self).__init__()
        self.opt = opt
        self.avg_pool = kwargs['avg_pool']
        self.recog_base = kwargs['recog_base']
        self.objectness_branch = kwargs['objectness_branch']
        self.box_reg_branch = kwargs['box_reg_branch']
        self.language_model = kwargs['llm']
        self.device = kwargs['device']

    def forward(self, input):
        roi_feats, roi_boxes, gt_boxes, gt_labels = input

        flat_roi_feats = torch.flatten(roi_feats, 1)

        roi_codes = self.recog_base(flat_roi_feats)
        objectness_scores = self.objectness_branch(roi_codes)

        pos_roi_codes = PosSlicer()([roi_codes, gt_labels])
        pos_roi_boxes = PosSlicer()([roi_boxes, gt_boxes])

        final_box_trans = self.box_reg_branch(pos_roi_codes)

        final_boxes = ApplyBoxTransform()(
            [pos_roi_boxes, final_box_trans])
        lm_output = self.language_model(pos_roi_codes, gt_labels)

        return [
            objectness_scores,
            pos_roi_boxes, final_box_trans, final_boxes,
            lm_output,
            gt_boxes, gt_labels,
        ]

class DenseCapModel(nn.Module):
    def __init__(self, opt):
        super(DenseCapModel, self).__init__()
        opt = opt or {}
        opt.cnn_name = utils.getopt(opt, 'cnn_name', 'vgg-16')
        opt.backend = utils.getopt(opt, 'backend', 'cudnn')
        opt.path_offset = utils.getopt(opt, 'path_offset', '')
        opt.dtype = utils.getopt(opt, 'dtype', 'torch.CudaTensor')
        opt.vocab_size = utils.getopt(opt, 'vocab_size')
        opt.std = utils.getopt(opt, 'std', 0.01)

        opt.final_nms_thresh = utils.getopt(opt, 'final_nms_thresh', 0.3)

        utils.ensureopt(opt, 'mid_box_reg_weight')
        utils.ensureopt(opt, 'mid_objectness_weight')
        utils.ensureopt(opt, 'end_box_reg_weight')
        utils.ensureopt(opt, 'end_objectness_weight')
        utils.ensureopt(opt, 'captioning_weight')

        opt.seq_length = utils.getopt(opt, 'seq_length')
        opt.rnn_encoding_size = utils.getopt(opt, 'rnn_encoding_size', 512)
        opt.rnn_size = utils.getopt(opt, 'rnn_size', 512)

        self.opt = opt
        self.device = opt['device']
        self.nets = {}

        self.net = nn.Sequential()

        cnn = net_utils.load_cnn(opt.cnn_name, self.device, opt.path_offset)


        if opt.cnn_name == 'vgg-16':
            conv_start1, conv_end1, conv_start2, conv_end2 = 0, 10, 10, 30
            fc_dim = 4096
            opt.input_dim = 512
            opt.output_height, opt.output_width = 7, 7
        else:
            raise ValueError(f'Unrecognized CNN "{opt.cnn_name}"')

        self.net.add_module('conv_net1', net_utils.subsequence(cnn, conv_start1, conv_end1))
        self.net.add_module('conv_net2', net_utils.subsequence(cnn, conv_start2, conv_end2))

        conv_full = net_utils.subsequence(cnn, conv_start1, conv_end2)
        x0, y0, sx, sy = net_utils.compute_field_centers(conv_full)
        self.opt.field_centers = [x0, y0, sx, sy]

        self.nets['localization_layer'] = LocalizationLayer(opt)
        self.net.add_module('localization_layer', self.nets['localization_layer'])

        self.nets['avg_pool'] = cnn.avgpool
        self.nets['recog_base'] = nn.Sequential()
        for i in range(6):
            self.nets['recog_base'].add_module('dnn_layer'+str(i), cnn.classifier[i])

        self.nets['objectness_branch'] = nn.Linear(fc_dim, 1)
        nn.init.normal_(self.nets['objectness_branch'].weight, mean=0, std=opt.std)
        nn.init.constant_(self.nets['objectness_branch'].bias, 0)

        self.nets['box_reg_branch'] = nn.Linear(fc_dim, 4)
        nn.init.constant_(self.nets['box_reg_branch'].weight, 0)
        nn.init.constant_(self.nets['box_reg_branch'].bias, 0)
        if opt['use_transformer']:
            self.nets['llm'] = Transformer(opt.vocab_size+3,
                                           opt.vocab_size+3,
                                           0,
                                           0,
                                           fc_dim,
                                           opt.idx_to_token,
                                           device=self.device)
        else:
            lm_opt = {
                'vocab_size': opt.vocab_size,
                'input_encoding_size': opt.rnn_encoding_size,
                'rnn_size': opt.rnn_size,
                'seq_length': opt.seq_length,
                'idx_to_token': opt.idx_to_token,
                'image_vector_dim': fc_dim,
                'device': self.device
            }
            self.nets['llm'] = LanguageModel(lm_opt)


        self._called_forward = True
        self.nets['device'] = self.device
        self.recog_net = RecogNet(self.opt, self.nets)

        self.net.add_module('recog_net', self.recog_net)
        self.crits = {}
        self.crits['objectness_crit'] = LogisticCriterion().to(self.device)
        self.crits['box_reg_crit'] = BoxRegressionCriterion(self.device, w=self.opt['end_box_reg_weight'])
        self.crits['lm_crit'] = nn.CTCLoss(zero_infinity=False).to(self.device)
        self.train()
        self.finetune_cnn = False

    def setTestArgs(self, kwargs):
        self.nets['localization_layer'].setTestArgs({'nms_thresh': utils.getopt(kwargs, 'rpn_nms_thresh', .7),
                                                    'max_proposals': utils.getopt(kwargs, 'num_proposals', 300)})
        self.opt['final_nms_thresh'] = utils.getopt(kwargs, 'final_nms_thresh', .3)

    def convert(self, dtype, use_cudnn=True):
        device = torch.device('cuda' if torch.cuda_is_available() and use_cudnn else 'cpu')
        self.to(dtype).to(device)

        if torch.cuda_is_available() and use_cudnn:
            torch.backends.cudnn.enabled = True
            self.nets['localization_layer'].nets['rpn'].to(dtype).to(device)

    def extractFeatures(self, input):
        assert input.dim() == 4 and input.size(0) == 1 and input.size(1) == 3
        H, W = input.size(2), input.size(3)
        self.nets['localization_layer'].setImageSize(H, W)

        output = self.net.forward(input)
        final_boxes_float = output[3].float()
        class_scores_float = output[0].float()
        boxes_scores = torch.FloatTensor(final_boxes_float.size(0), 5)
        boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
        boxes_scores[:, :4].copy_(boxes_x1y1x2y2)
        boxes_scores[:, 4].copy_(class_scores_float[:, 0])
        idx = box_utils.nms(boxes_scores, self.opt['final_nms_thresh'])

        boxes_xcycwh = final_boxes_float.index_select(0, idx).type_as(self.output[3])
        feats = self.nets['recog_base'].output.float().index_select(0, idx).type_as(self.output[3])

        return boxes_xcycwh, feats

    def forward_test(self, input):
        self.eval()
        with torch.no_grad():
            assert input.dim() == 4 and input.size(0) == 1 and input.size(1) == 3
            H, W = input.size(2), input.size(3)
            self.nets['localization_layer'].setImageSize(H, W)
            self.output = self.net.forward(input)
            if self.training is False and self.opt['final_nms_thresh']>0:
                final_boxes_float = self.output[3].float()
                class_scores_float = self.output[0].float().squeeze(1)
                lm_output_float = self.output[4].float()
                boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
                idx = torchvision.ops.nms(boxes_x1y1x2y2, class_scores_float, self.opt['final_nms_thresh'])
                self.output[3] = final_boxes_float.index_select(0, idx).type_as(self.output[3])
                self.output[0] = class_scores_float.index_select(0, idx).type_as(self.output[0])
                self.output[4] = lm_output_float.index_select(0, idx).type_as(self.output[4])
            final_boxes = self.output[3]
            objectness_scores = self.output[0]
            captions = self.output[4]
            captions = self.nets['llm'].decode_sequence(captions)
        return final_boxes, objectness_scores, captions


    def setGroundTruth(self, gt_boxes, gt_labels):
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self._called_forward = False
        self.nets['localization_layer'].setGroundTruth(gt_boxes, gt_labels)

    def getParameters(self):
        cnn_params = self.net.conv_net2.parameters()
        fakenet = nn.Sequential()
        for layer in self.net.modules():
            fakenet.append(layer)
        params = fakenet.parameters()
        return params, cnn_params


    def clearState(self):
        self.net.clearState()
        for k, v in self.crits.items():
            if v.clearState:
                v.clearState()


    def forward_backward(self, data):

        if self.training is True:
            self.setGroundTruth(data.gt_boxes.to(self.device), data.gt_labels.to(self.device))
            assert data.image.dim() == 4 and data.image.size(0) == 1 and data.image.size(1) == 3
            H, W = data.image.size(2), data.image.size(3)
            self.nets['localization_layer'].setImageSize(H, W)
            self.nets['localization_layer'].eval_mode = False
            assert not self._called_forward, 'Must call setGroundTruth before training-time forward pass'
            self._called_forward = True
            self.out = self.net.forward(data.image)

            objectness_scores = self.out[0]
            pos_roi_boxes = self.out[1]
            final_box_trans = self.out[2]
            lm_output = self.out[4]
            gt_boxes = self.out[5]
            gt_labels = self.out[6]

            num_boxes = objectness_scores.size(0)
            num_pos = pos_roi_boxes.size(0)

            objectness_pos_labels = torch.ones(num_pos, dtype=torch.long)
            objectness_labels = torch.concat((objectness_pos_labels, torch.zeros(num_boxes - num_pos, dtype=torch.long)), dim=0)
            end_objectness_loss = self.crits['objectness_crit'].forward(objectness_scores, objectness_labels)

            end_objectness_loss = end_objectness_loss * self.opt['end_objectness_weight']

            end_box_reg_loss = self.crits['box_reg_crit'].forward([pos_roi_boxes, final_box_trans], gt_boxes)

            gt_targets = self.nets['llm'].get_target(gt_labels, True)
            captioning_loss = CustomCrossEntropyLoss().forward(lm_output, gt_targets)
            captioning_loss = captioning_loss * self.opt['captioning_weight']
            ll_losses = self.nets['localization_layer'].stats['losses']
            ll_obj_loss = ll_losses.obj_loss
            ll_reg_loss = ll_losses.box_reg_loss
            losses = {
                'mid_objectness_loss': ll_obj_loss,
                'mid_box_reg_loss': ll_reg_loss,
                'end_objectness_loss': end_objectness_loss,
                'end_box_reg_loss': end_box_reg_loss,
                'captioning_loss': captioning_loss,
            }
            total_loss = 0
            for k, v in losses.items():
                total_loss = total_loss + v
            losses['total_loss'] = total_loss
            total_loss.backward()

        else:
            with torch.no_grad():
                self.setGroundTruth(data.gt_boxes.to(self.device), data.gt_labels.to(self.device))
                assert data.image.dim() == 4 and data.image.size(0) == 1 and data.image.size(1) == 3
                H, W = data.image.size(2), data.image.size(3)
                self.nets['localization_layer'].setImageSize(H, W)
                assert not self._called_forward, 'Must call setGroundTruth before training-time forward pass'
                self._called_forward = True
                self.nets['localization_layer'].eval_mode = True
                self.out = self.net.forward(data.image)

                objectness_scores = self.out[0]
                pos_roi_boxes = self.out[1]
                final_box_trans = self.out[2]
                lm_output = self.out[4]
                gt_boxes = self.out[5]
                gt_labels = self.out[6]

                num_boxes = objectness_scores.size(0)
                num_pos = pos_roi_boxes.size(0)

                objectness_pos_labels = torch.ones(num_pos, dtype=torch.long)
                objectness_labels = torch.concat(
                    (objectness_pos_labels, torch.zeros(num_boxes - num_pos, dtype=torch.long)), dim=0)
                end_objectness_loss = self.crits['objectness_crit'].forward(objectness_scores, objectness_labels)

                end_objectness_loss = end_objectness_loss * self.opt['end_objectness_weight']

                end_box_reg_loss = self.crits['box_reg_crit'].forward([pos_roi_boxes, final_box_trans], gt_boxes)

                gt_targets = self.nets['llm'].get_target(gt_labels, True)
                captioning_loss = CustomCrossEntropyLoss().forward(lm_output, gt_targets)
                captioning_loss = captioning_loss * self.opt['captioning_weight']
                ll_losses = self.nets['localization_layer'].stats['losses']
                ll_obj_loss = ll_losses.obj_loss
                ll_reg_loss = ll_losses.box_reg_loss
                losses = {
                    'mid_objectness_loss': ll_obj_loss,
                    'mid_box_reg_loss': ll_reg_loss,
                    'end_objectness_loss': end_objectness_loss,
                    'end_box_reg_loss': end_box_reg_loss,
                    'captioning_loss': captioning_loss,
                }
                total_loss = 0
                for k, v in losses.items():
                    total_loss = total_loss + v
                losses['total_loss'] = total_loss

        return losses
