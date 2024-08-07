import torchvision
import torch
import torch.nn as nn


class ResnetCapModel(nn.Module):
    def __init__(self, opt):
        super(ResnetCapModel, self).__init__()
        self.opt = opt
        self.device = opt['device']
        self.crit = nn.BCEWithLogitsLoss()
        self.attr_names = opt.attributes_labels
        self.test_threshold = 0.75
        # network architecture


        self.net = nn.Sequential()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.net.add_module('resnet_backbone', self.features)
        self.net.add_module('flat_layer', nn.Flatten())
        self.net.add_module('attribute_layer', nn.Sequential(nn.Linear(2048, 2048), nn.Linear(2048, 40)))

    def forward(self, data):
        assert data.image.dim() == 4 and data.image.size(0) == 1 and data.image.size(1) == 3
        output = self.net.forward(data.image)
        loss = self.crit(output, data.gt_attr.to(self.device).float())
        loss.backward()
        return loss

    def prediction(self, data, do_eval=True):
        with torch.no_grad():
            loss = None
            assert data.image.dim() == 4 and data.image.size(0) == 1 and data.image.size(1) == 3
            output = self.net.forward(data.image)
            if do_eval:
                loss = self.crit(output, data.gt_attr.to(self.device))
            else:
                output = self.construct_sentence(output)
        return loss, output

    def construct_sentence(self, probs):
        seq = 'a '
        attributes = torch.gt(probs, self.test_threshold)

        for i in range(5):
            if attributes[i]:
                seq = seq + self.attr_names[i] + ', '

        if attributes[5]:
            seq = seq + self.attr_names[5]
        else:
            seq = seq + 'Woman'

        for i in range(6, 35):
            if attributes[i]:
                seq = seq + ', with ' + self.attr_names[i]

        for i in range(35, len(self.attr_names)):
            if attributes[i]:
                seq = seq + ', ' + self.attr_names[i]

        return seq



