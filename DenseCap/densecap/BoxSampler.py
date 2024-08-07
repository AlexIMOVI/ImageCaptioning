import torch
import torch.nn as nn
import torchvision
import box_utils
from densecap_utils import utils

class BoxSampler(nn.Module):
    def __init__(self, opt):
        super(BoxSampler, self).__init__()
        self.low_thresh = utils.getopt(opt, 'low_thresh', 0.3)
        self.high_thresh = utils.getopt(opt, 'high_thresh', 0.7)
        self.batch_size = utils.getopt(opt, 'batch_size', 256)
        self.to(opt['device'])
    def setBounds(self, bounds):
        self.x_min = bounds.x_min
        self.x_max = bounds.x_max
        self.y_min = bounds.y_min
        self.y_max = bounds.y_max

    def forward(self, input):
        input_boxes, target_boxes = input
        input_x1y1 = box_utils.xcycwh_to_x1y1x2y2(input_boxes).squeeze(0)
        target_x1y1 = box_utils.xcycwh_to_x1y1x2y2(target_boxes).squeeze(0)
        ious = torchvision.ops.box_iou(input_x1y1, target_x1y1)

        input_max_iou, input_idx = torch.max(ious, dim=1)  # N x B1
        target_max_iou, target_idx = torch.max(ious, dim=0)  # N x B2

        self.pos_mask = torch.gt(input_max_iou, self.high_thresh)  # N x B1
        self.neg_mask = torch.lt(input_max_iou, self.low_thresh)  # N x B1

        if self.x_min is not None and self.y_min is not None and self.x_max is not None and self.y_max is not None:
            boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(input_boxes)
            x_min_mask = torch.lt(boxes_x1y1x2y2[0, :, 0], self.x_min)
            y_min_mask = torch.lt(boxes_x1y1x2y2[0, :, 1], self.y_min)
            x_max_mask = torch.gt(boxes_x1y1x2y2[0, :, 2], self.x_max)
            y_max_mask = torch.gt(boxes_x1y1x2y2[0, :, 3], self.y_max)

            self.pos_mask[torch.stack((x_min_mask, y_min_mask, x_max_mask, y_max_mask), dim=1).any(dim=1)] = 0
            self.neg_mask[torch.stack((x_min_mask, y_min_mask, x_max_mask, y_max_mask), dim=1).any(dim=1)] = 0

        # self.pos_mask[:, target_idx] = 1
        # self.neg_mask[:, target_idx] = 0
        self.pos_mask[target_idx] = 1
        self.neg_mask[target_idx] = 0

        # self.pos_mask = self.pos_mask.view(-1).byte()
        # self.neg_mask = self.neg_mask.view(-1).byte()
        self.pos_mask = self.pos_mask.byte()
        self.neg_mask = self.neg_mask.byte()

        if self.neg_mask.sum() == 0:
            self.neg_mask = self.neg_mask.mul(-self.pos_mask).add(1)

        pos_mask_nonzero = self.pos_mask.nonzero().view(-1)
        neg_mask_nonzero = self.neg_mask.nonzero().view(-1)

        total_pos = pos_mask_nonzero.size(0)
        total_neg = neg_mask_nonzero.size(0)

        num_pos = min(self.batch_size // 2, total_pos)
        num_neg = self.batch_size - num_pos

        pos_p = torch.ones(total_pos)
        pos_sample_idx = torch.multinomial(pos_p, num_pos, False)

        neg_p = torch.ones(total_neg)
        neg_replace = total_neg < num_neg
        neg_sample_idx = torch.multinomial(neg_p, num_neg, neg_replace)

        pos_input_idx = pos_mask_nonzero[pos_sample_idx]
        pos_target_idx = input_idx[pos_input_idx]
        neg_input_idx = neg_mask_nonzero[neg_sample_idx]

        self.output = [pos_input_idx, pos_target_idx, neg_input_idx]
        return self.output



