import torch
import numpy as np
from BoxSampler import BoxSampler

class BoxSamplerHelper(torch.nn.Module):
    def __init__(self, opt):
        super(BoxSamplerHelper, self).__init__()

        self.box_sampler = BoxSampler(opt)
        self.device = opt['device']
        self.output = [[], [], []]
        self.num_pos, self.num_neg = None, None
        self.pos_input_idx = None
        self.pos_target_idx = None
        self.neg_input_idx = None

        self.to(self.device)

    def setBounds(self, bounds):
        self.box_sampler.setBounds(bounds)

    def forward(self, input):

        input_data, target_data = input
        input_boxes = input_data[0]
        target_boxes = target_data[0]
        N = input_boxes.size(0)
        assert N == 1, 'Only minibatches of 1 are supported'

        self.pos_input_idx, self.pos_target_idx, self.neg_input_idx = self.box_sampler.forward([input_boxes, target_boxes])

        self.num_pos = self.pos_input_idx.size(0)
        self.num_neg = self.neg_input_idx.size(0)

        output_boxes_pos = torch.index_select(input_boxes,1,self.pos_input_idx).squeeze(0)
        output_boxes_neg = torch.index_select(input_boxes,1,self.neg_input_idx).squeeze(0)
        output_boxes = [output_boxes_pos, output_boxes_neg]
        output_anchors_pos = torch.index_select(input_data[1],1,self.pos_input_idx).squeeze(0)
        output_anchors_neg = torch.index_select(input_data[1],1,self.neg_input_idx).squeeze(0)
        output_anchors = [output_anchors_pos, output_anchors_neg]
        output_trans_pos = torch.index_select(input_data[2], 1, self.pos_input_idx).squeeze(0)
        output_trans_neg = torch.index_select(input_data[2], 1, self.neg_input_idx).squeeze(0)
        output_trans = [output_trans_pos, output_trans_neg]
        output_scores_pos = torch.index_select(input_data[3], 1, self.pos_input_idx).squeeze(0)
        output_scores_neg = torch.index_select(input_data[3], 1, self.neg_input_idx).squeeze(0)
        output_scores = [output_scores_pos, output_scores_neg]

        output_target_boxes = torch.index_select(target_boxes, 1, self.pos_target_idx).squeeze(0)
        output_target_labels = torch.index_select(target_data[1], 1, self.pos_target_idx).squeeze(0)
        output_target = [output_target_boxes, output_target_labels]

        self.output = [output_boxes, output_anchors, output_trans, output_scores, output_target]
        return self.output
