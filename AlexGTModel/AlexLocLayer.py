import torch
from AlexGTModel.BilinearRoiPooling import BilinearRoiPooling
from AlexGTModel.densecap_utils import utils
import torch.nn as nn

class LocalizationLayer(nn.Module):
    def __init__(self, opt=None):
        super(LocalizationLayer, self).__init__()
        # Defaults
        opt = opt or {}
        opt.output_height = utils.getopt(opt, 'output_height')
        opt.output_width = utils.getopt(opt, 'output_width')
        self.opt = opt
        self.roi_pooling = BilinearRoiPooling(self.opt.output_height, self.opt.output_width, self.opt['device'])
        self.device = opt['device']
        self.eval_mode = False
    def setImageSize(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self._called_forward_size = False
        self._called_backward_size = False


    def setGroundTruth(self, gt_boxes):
        self.gt_boxes = gt_boxes.to(self.opt['device'])
        self._called_forward_gt = False
        self._called_backward_gt = False

    def forward(self, input):
        if self.eval_mode:
            with torch.no_grad():
                self.cnn_features = input
                assert (
                            self.gt_boxes is not None and not self._called_forward_gt), 'must call setgroundtruth before training-time forward pass'
                gt_boxes = self.gt_boxes
                self._called_forward_gt = True

                assert (
                            self.image_height is not None and self.image_width is not None and not self._called_forward_size), 'Must call setImageSize before each forward pass'
                self._called_forward_size = True

                N = self.cnn_features.size(0)
                assert N == 1, 'Only minibatches with N = 1 are supported'
                B1 = gt_boxes.size(1)
                assert (gt_boxes.dim() == 3 and gt_boxes.size(0) == N and gt_boxes.size(
                    2) == 4), 'gt_boxes must have shape (N, B1, 4)'

                self.roi_pooling.setImageSize(self.image_height, self.image_width)
                # img_box = torch.Tensor(
                #     [[self.image_width // 2, self.image_height // 2, self.image_width, self.image_height]]).to(
                #     self.device)
                self.roi_features = self.roi_pooling.forward([self.cnn_features[0], gt_boxes[0]])

                self.output = torch.flatten(self.roi_features.to(self.device), 1)
        else:
            self.cnn_features = input
            assert (
                    self.gt_boxes is not None and not self._called_forward_gt), 'must call setgroundtruth before training-time forward pass'
            gt_boxes = self.gt_boxes
            self._called_forward_gt = True

            assert (
                    self.image_height is not None and self.image_width is not None and not self._called_forward_size), 'Must call setImageSize before each forward pass'
            self._called_forward_size = True

            N = self.cnn_features.size(0)
            assert N == 1, 'Only minibatches with N = 1 are supported'
            B1 = gt_boxes.size(1)
            assert (gt_boxes.dim() == 3 and gt_boxes.size(0) == N and gt_boxes.size(
                2) == 4), 'gt_boxes must have shape (N, B1, 4)'

            self.roi_pooling.setImageSize(self.image_height, self.image_width)
            # img_box = torch.Tensor(
            #     [[self.image_width // 2, self.image_height // 2, self.image_width, self.image_height]]).to(self.device)
            self.roi_features = self.roi_pooling.forward([self.cnn_features[0], gt_boxes[0]])

            self.output = torch.flatten(self.roi_features.to(self.device), 1)
        return self.output