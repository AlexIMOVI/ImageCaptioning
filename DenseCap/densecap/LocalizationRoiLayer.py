from easydict import EasyDict as edict
import torchvision
import box_utils
from LogisticCriterion import LogisticCriterion
from BoxRegressionCriterion import BoxRegressionCriterion
from BilinearRoiPooling import BilinearRoiPooling
from ReshapeBoxFeatures import ReshapeBoxFeatures
from ApplyBoxTransform import ApplyBoxTransform
from BoxSamplerHelper import BoxSamplerHelper
from RegularizeLayer import RegularizeLayer
from MakeAnchors import MakeAnchors
from box_utils import clip_boxes
from densecap_utils import utils
import torch
import torch.nn as nn


class BuildRPNRoi(nn.Module):
    def __init__(self, opt):
        super(BuildRPNRoi, self).__init__()
        self.opt = opt
        self.anchors = opt.get('anchors', None)
        if self.anchors is None:
            self.anchors = torch.Tensor([
                [45, 90], [90, 45], [64, 64],
                [90, 180], [180, 90], [128, 128],
                [181, 362], [362, 181], [256, 256],
                [362, 724], [724, 362], [512, 512],
            ]).t().clone().to(opt['device'])
            self.anchors = self.anchors * opt.anchor_scale
        self.num_anchors = self.anchors.size(1)
        self.rpn_base = nn.Sequential()

        # Add an extra conv layer and a ReLU
        pad = opt.rpn_filter_size // 2
        conv_layer = nn.Conv2d(
            opt['input_dim'], opt['rpn_num_filters'],
            opt['rpn_filter_size'], padding=pad)

        nn.init.normal_(conv_layer.weight, mean=0, std=opt['std'])
        nn.init.constant_(conv_layer.bias, 0)
        self.rpn_base.add_module('conv_layer', conv_layer)
        self.rpn_base.add_module('ReLU', nn.ReLU(inplace=True))

        self.box_branch = nn.Sequential()
        box_conv_layer = nn.Conv2d(
            opt['rpn_num_filters'], 4 * self.num_anchors, 1)
        if opt['zero_box_conv']:
            nn.init.constant_(box_conv_layer.weight, 0)
        else:
            nn.init.normal_(box_conv_layer.weight, mean=0, std=opt['std'])
        nn.init.constant_(box_conv_layer.bias, 0)

        self.box_branch.add_module('box_conv_layer', box_conv_layer)
        self.box_branch.add_module('RegularizeLayer', RegularizeLayer(opt['box_reg_decay']))
        self.seq = nn.Sequential(
            MakeAnchors(*opt['field_centers'], self.anchors),
            ReshapeBoxFeatures(self.num_anchors)
        )
        self.reshapeboxfeatures = ReshapeBoxFeatures(self.num_anchors)
        self.applyboxtransform = ApplyBoxTransform()

        # Branch to produce box / not box scores for each anchor
        self.rpn_branch = nn.Sequential()
        rpn_conv_layer = nn.Conv2d(
            opt['rpn_num_filters'], 1 * self.num_anchors, 1)

        nn.init.normal_(rpn_conv_layer.weight, mean=0, std=opt['std'])
        nn.init.constant_(rpn_conv_layer.bias, 0)
        self.rpn_branch.add_module('rpn_conv_layer', rpn_conv_layer)
        self.rpn_branch.add_module('ReshapeBoxFeatures', ReshapeBoxFeatures(self.num_anchors))

        self.to(opt['device'])

    def forward(self, features):

        self.out_rpn_base = self.rpn_base(features)

        out_rpn_branch = self.rpn_branch(self.out_rpn_base)

        self.out_box_base = self.box_branch(self.out_rpn_base)

        out_seq = self.seq(self.out_box_base)

        out_rbf = self.reshapeboxfeatures(self.out_box_base)

        out_apply_box = self.applyboxtransform.forward([out_seq, out_rbf])

        return out_apply_box, out_seq, out_rbf, out_rpn_branch


class LocalizationRoiLayer(nn.Module):
    def __init__(self, opt=None):
        super(LocalizationRoiLayer, self).__init__()
        # Defaults
        opt = opt or {}
        opt.input_dim = utils.getopt(opt,'input_dim')
        opt.output_height = utils.getopt(opt, 'output_height')
        opt.output_width = utils.getopt(opt, 'output_width')

        # Field centers
        opt.field_centers = utils.getopt(opt, 'field_centers')

        opt.backend = utils.getopt(opt, 'backend', 'cudnn')
        opt.rpn_filter_size = utils.getopt(opt, 'rpn_filter_size', 3)
        opt.rpn_num_filters = utils.getopt(opt, 'rpn_num_filters', 256)
        opt.zero_box_conv = utils.getopt(opt, 'zero_box_conv', True)
        opt.std = utils.getopt(opt, 'std', 0.01)
        opt.anchor_scale = utils.getopt(opt, 'anchor_scale', 1.0)

        opt.sampler_batch_size = utils.getopt(opt, 'sampler_batch_size', 256)
        opt.sampler_high_thresh = utils.getopt(opt, 'sampler_high_thresh', 0.7)
        opt.sampler_low_thresh = utils.getopt(opt, 'sampler_low_thresh', 0.5)
        opt.train_remove_outbounds_boxes = utils.getopt(opt, 'train_remove_outbounds_boxes', 1)

        utils.ensureopt(opt,'mid_box_reg_weight')
        utils.ensureopt(opt,'mid_objectness_weight')

        opt.box_reg_decay = utils.getopt(opt, 'box_reg_decay', 0)
        self.opt = opt
        self.losses = {}
        self.device = utils.getopt(opt, 'device')
        self.nets = {}

        self.rpn = BuildRPNRoi(opt)

        self.box_sampler_helper = BoxSamplerHelper(opt)

        self.roi_pooling = BilinearRoiPooling(self.opt.output_height, self.opt.output_width, self.opt['device'])

        self.nets['obj_crit'] = LogisticCriterion().to(self.device)  # for objectness
        self.nets['box_reg_crit'] = BoxRegressionCriterion(self.device, w=self.opt['mid_box_reg_weight'])
        self.reset_stats()
        self.setTestArgs()
        self.train()

    def reset_stats(self):
        self.stats = {}
        self.stats['losses'] = edict()
        self.stats['times'] = edict()
        self.stats['vars'] = edict()
    def setImageSize(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self._called_forward_size = False
        self._called_backward_size = False

    def setGroundTruth(self, gt_boxes, gt_labels):
        self.gt_boxes = gt_boxes.to(self.opt['device'])
        self.gt_labels = gt_labels.to(self.opt['device'])
        self._called_forward_gt = False
        self._called_backward_gt = False

    def setTestArgs(self, args=None):
        args = args or {}
        self.test_clip_boxes = utils.getopt(args, 'clip_boxes', True)
        self.test_nms_thresh = utils.getopt(args, 'nms_thresh', 0.7)
        self.test_max_proposals = utils.getopt(args, 'max_proposals', 300)

    def forward(self, input):
        if self.training is True:
            return self._forward_train(input)
        else:
            return self._forward_test(input)

    def _forward_test(self, input):
        cnn_features = input
        arg = {
            'clip_boxes': self.test_clip_boxes,
            'nms_thresh': self.test_nms_thresh,
            'max_proposals': self.test_max_proposals
        }
        assert (self.image_height and self.image_width and not self._called_forward_size), 'must call setimagesize before each forward pass'
        self._called_forward_size = True

        rpn_out = self.rpn.forward(cnn_features)

        rpn_boxes, rpn_anchors = rpn_out[0], rpn_out[1]
        rpn_trans, rpn_scores = rpn_out[2], rpn_out[3]
        num_boxes = rpn_boxes.size(1)

        if arg['clip_boxes']:
            bounds = edict()
            bounds.x_min = 0
            bounds.y_min = 0
            bounds.x_max = self.image_width - 1
            bounds.y_max = self.image_height - 1
            rpn_boxes, valid = clip_boxes(rpn_boxes, bounds, 'xcycwh')

            def clamp_data(data):
                assert data.size(0) == 1, 'must have 1 image per batch'
                assert data.dim() == 3
                return data[:, valid, :]

            rpn_boxes = clamp_data(rpn_boxes)
            rpn_anchors = clamp_data(rpn_anchors)
            rpn_trans = clamp_data(rpn_trans)
            rpn_scores = clamp_data(rpn_scores)

            num_boxes = rpn_boxes.size(1)

        rpn_boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(rpn_boxes)

        verbose = False
        if verbose:
            print(f'in localizationlayer forward_test')
            print(f'before NMS : {num_boxes}')
            print(f'using nms threshold {arg["nms_thresh"]}')

        if arg['max_proposals'] == -1:
            idx = torchvision.ops.nms(rpn_boxes_x1y1x2y2[0], rpn_scores[0, :, 0], arg['nms_thresh'])
        else:
            idx = torchvision.ops.nms(rpn_boxes_x1y1x2y2[0], rpn_scores[0, :, 0], arg['nms_thresh'])[:arg['max_proposals']]

        rpn_boxes_nms = rpn_boxes.index_select(1, idx)[0]

        if verbose:
            print(f'After nms there is {rpn_boxes_nms.size(0)} boxes')

        self.roi_pooling.setImageSize(self.image_height, self.image_width)
        roi_features = self.roi_pooling.forward([cnn_features[0], rpn_boxes_nms])

        empty = roi_features.new()
        self.output = [roi_features, rpn_boxes_nms, empty, empty]

        return self.output

    def _forward_train(self, input):

        self.cnn_features = input
        assert (self.gt_boxes is not None and self.gt_labels is not None and not self._called_forward_gt), 'must call setgroundtruth before training-time forward pass'
        gt_boxes, gt_labels = self.gt_boxes, self.gt_labels
        self._called_forward_gt = True

        assert (self.image_height is not None and self.image_width is not None and not self._called_forward_size), 'Must call setImageSize before each forward pass'
        self._called_forward_size = True

        N = self.cnn_features.size(0)
        assert N == 1, 'Only minibatches with N = 1 are supported'
        B1 = gt_boxes.size(1)
        assert (gt_boxes.dim() == 3 and gt_boxes.size(0) == N and gt_boxes.size(2) == 4), 'gt_boxes must have shape (N, B1, 4)'
        assert (gt_labels.dim() == 3 and gt_labels.size(0) == N and gt_labels.size(1) == B1), 'gt_labels must have shape (N, B1, L)'

        self.rpn_boxes, self.rpn_anchors, self.rpn_trans, self.rpn_scores = self.rpn.forward(self.cnn_features)
        self.rpn_out = [self.rpn_boxes, self.rpn_anchors, self.rpn_trans, self.rpn_scores]

        if self.opt['train_remove_outbounds_boxes'] == 1:
            bounds = edict()
            bounds.x_min = 0
            bounds.y_min = 0
            bounds.x_max = self.image_width - 1
            bounds.y_max = self.image_height - 1

            self.box_sampler_helper.setBounds(bounds)

        sampler_out = self.box_sampler_helper.forward([self.rpn_out, [gt_boxes, gt_labels]])

        self.sampled_boxes = sampler_out[0]
        self.sampled_anchors = sampler_out[1]
        self.sampled_trans = sampler_out[2]
        self.sampled_scores = sampler_out[3]
        self.sampled_targets = sampler_out[4]
        self.pos_boxes, self.pos_anchors = self.sampled_boxes[0], self.sampled_anchors[0]
        self.pos_trans, self.pos_scores = self.sampled_trans[0], self.sampled_scores[0]

        self.pos_target_boxes, self.pos_target_labels = self.sampled_targets[0], self.sampled_targets[1]

        self.neg_boxes = self.sampled_boxes[1]
        self.neg_scores = self.sampled_scores[1]

        self.scores = torch.cat((self.pos_scores, self.neg_scores), dim=0)
        num_pos, num_neg = self.pos_scores.size(0), self.neg_scores.size(0)
        self.roi_boxes = torch.Tensor(num_pos + num_neg, 4).to(self.device)
        self.roi_boxes[:num_pos].copy_(self.pos_boxes)
        self.roi_boxes[num_pos:num_neg+num_pos].copy_(self.neg_boxes)

        self.roi_pooling.setImageSize(self.image_height, self.image_width)
        self.roi_features = self.roi_pooling.forward([self.cnn_features[0], self.roi_boxes])

        objectness_pos_labels = torch.ones(num_pos, dtype=torch.long)
        objectness_labels = torch.concat((objectness_pos_labels, torch.zeros(num_neg, dtype=torch.long)),
                                         dim=0)
        obj_weight = self.opt['mid_objectness_weight']
        mid_objectness_loss = self.nets['obj_crit'].forward(self.scores, objectness_labels) * obj_weight

        self.stats['losses'].obj_loss = mid_objectness_loss

        loss = self.nets['box_reg_crit'].forward([self.pos_anchors, self.pos_trans], self.pos_target_boxes)
        self.stats['losses'].box_reg_loss = loss

        reg_mods = self.rpn.box_branch.RegularizeLayer
        self.stats['losses'].box_decay_loss = reg_mods.loss

        # ll_loss = loss + mid_objectness_loss  # + reg_mods.loss
        # ll_loss.backward()


        self.output = [self.roi_features.detach().to(self.device), self.roi_boxes.detach().to(self.device), self.pos_target_boxes.detach().to(self.device), self.pos_target_labels.detach().to(self.device)]
        return self.output
