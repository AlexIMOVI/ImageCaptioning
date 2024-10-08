import torch
import torch.nn as nn
import torch.nn.functional as F
from BoxToAffine import BoxToAffine
from BatchBilinearSamplerBHWD import BatchBilinearSamplerBHWD, AffineGridGeneratorBHWD

class PermuteLayer(nn.Module):
    def __init__(self,dims):
        super(PermuteLayer,self).__init__()
        self.dims = dims
    def forward(self, inputs):
        output = inputs.transpose(self.dims[0],self.dims[1])
        return output.transpose(self.dims[2], self.dims[3])
class BilinearRoiPooling(nn.Module):
    """
    BilinearRoiPooling is a layer that uses bilinear sampling to pool features for a
    region of interest (RoI) into a fixed size.
    """

    def __init__(self, height, width, device):
        super(BilinearRoiPooling, self).__init__()
        self.height = height
        self.width = width
        self.box_branch = nn.Sequential()
        self.box_to_affine = BoxToAffine()
        self.box_branch.append(self.box_to_affine)
        self.box_branch.append(AffineGridGeneratorBHWD(self.height, self.width, device))
        #self.perm_layer1 = PermuteLayer([0,2,1,3])
        self.bbsBHWD = BatchBilinearSamplerBHWD()
        #self.perm_layer2 = PermuteLayer([2,3,3,3])

        self.image_height = None
        self.image_width = None
        self.called_forward = False
        self.called_backward = False

    def setImageSize(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.called_forward = False
        self.called_backward = False
        return self

    def forward(self, input):
        assert self.image_height is not None and self.image_width is not None and not self.called_forward, \
            'Must call setImageSize before each forward pass'
        self.box_to_affine.setSize(self.image_height, self.image_width)
        features, boxes = input

        self.out_box_branch = self.box_branch.forward(boxes)

        out_bbs = self.bbsBHWD.forward([features, self.out_box_branch])

        self.called_forward = True
        return out_bbs
