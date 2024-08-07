import torch
import torch.nn as nn
import box_utils

class BoxIoU(nn.Module):
    def __init__(self, device):
        super(BoxIoU, self).__init__()
        self.area1 = None
        self.area2 = None
        self.overlap = None
        self.device = device

    def forward(self, input):
        box1 = input[0]
        box2 = input[1]
        N, B1, B2 = box1.size(0), box1.size(1), box2.size(1)
        self.area1 = torch.mul(box1[:, :, 2], box1[:, :, 3])
        self.area2 = torch.mul(box2[:, :, 2], box2[:, :, 3])
        area1_expand = self.area1.view(N, B1, 1).expand(N, B1, B2)
        area2_expand = self.area2.view(N, 1, B2).expand(N, B1, B2)

        convert_boxes = box_utils.xcycwh_to_x1y1x2y2
        box1_lohi = convert_boxes(box1)  # N x B1 x 4
        box2_lohi = convert_boxes(box2)  # N x B2 x 4
        box1_lohi_expand = box1_lohi.view(N, B1, 1, 4).expand(N, B1, B2, 4)
        box2_lohi_expand = box2_lohi.view(N, 1, B2, 4).expand(N, B1, B2, 4)

        x0 = torch.max(box1_lohi_expand[:, :, :, 0],
                       box2_lohi_expand[:, :, :, 0])
        y0 = torch.max(box1_lohi_expand[:, :, :, 1],
                       box2_lohi_expand[:, :, :, 1])
        x1 = torch.min(box1_lohi_expand[:, :, :, 2],
                       box2_lohi_expand[:, :, :, 2])
        y1 = torch.min(box1_lohi_expand[:, :, :, 3],
                       box2_lohi_expand[:, :, :, 3])

        w = (x1 - x0).clamp(0)
        h = (y1 - y0).clamp(0)

        intersection = torch.mul(w, h).to(self.device)
        self.output = intersection / (area1_expand + area2_expand - intersection)

        return self.output
