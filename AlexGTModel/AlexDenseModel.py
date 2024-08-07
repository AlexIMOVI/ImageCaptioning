import torchvision
import torch
import torch.nn as nn
from AlexGTModel.AlexTransformer import Transformer
from AlexGTModel.LSTMLoss import CustomCrossEntropyLoss
from AlexGTModel.AlexLocLayer import LocalizationLayer
from AlexGTModel.AlexDenseLangage import LanguageModel

class AlexCapModel(nn.Module):
    def __init__(self, opt):
        super(AlexCapModel, self).__init__()
        self.opt = opt
        self.device = opt['device']

        # network architecture
        if opt.use_dropout:
            self.dropout = opt.drop_value
        else:
            self.dropout = 0
        self.net = nn.Sequential()

        opt.output_height, opt.output_width = 7, 7
        cnn = torchvision.models.vgg16(weights=torchvision.models.vgg.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(cnn.features.children())[:-1])
        self.net.add_module('vgg16_backbone', self.features)
        self.loc_layer = LocalizationLayer(opt)
        self.net.add_module('loc_layer', self.loc_layer)
        self.classifier = nn.Sequential(*list(cnn.classifier.children())[:-1])
        self.net.add_module('full_conv', self.classifier)
        fc_dim = 4096
        if opt.use_lstm:
            self.llm = LanguageModel(opt.vocab_size,
                                     512,
                                     fc_dim,
                                     512,
                                     opt.seq_length,
                                     opt.num_layers,
                                     opt.idx_to_token,
                                     self.dropout,
                                     self.device,
                                     opt.use_curriculum_learning)
        else:
            self.llm = Transformer(src_vocab_size=opt.vocab_size + 3,
                                   fc_dim=fc_dim,
                                   token_dict=opt.idx_to_token,
                                   max_length=opt.seq_length+1,
                                   device=self.device)

        # criteria
        self.crit = CustomCrossEntropyLoss()
        self.llm.use_curriculum = opt.use_curriculum_learning
        self.eval_mode = False
        self.train()

    def set_eval(self, value):
        self.eval_mode = value
        self.loc_layer.eval_mode = value

    def forward_train(self, data):
        gt_labels = data.gt_labels[0].to(self.device)
        if self.eval_mode:
            with torch.no_grad():
                assert data.image.dim() == 4 and data.image.size(0) == 1 and data.image.size(1) == 3
                self.loc_layer.setGroundTruth(data.gt_boxes.to(self.device))
                H, W = data.image.size(2), data.image.size(3)
                self.loc_layer.setImageSize(H, W)
                self.features_ouput = self.net(data.image)
                out_llm = self.llm(self.features_ouput, gt_labels)
                gt_targets = self.llm.get_target(gt_labels, make_target=True)
                captioning_loss = self.crit(out_llm, gt_targets)
        else:
            assert data.image.dim() == 4 and data.image.size(0) == 1 and data.image.size(1) == 3
            self.loc_layer.setGroundTruth(data.gt_boxes.to(self.device))
            H, W = data.image.size(2), data.image.size(3)
            self.loc_layer.setImageSize(H, W)
            self.features_ouput = self.net(data.image)
            out_llm = self.llm(self.features_ouput, gt_labels)
            gt_targets = self.llm.get_target(gt_labels, make_target=True)
            captioning_loss = self.crit(out_llm, gt_targets)
            captioning_loss.backward()
        return captioning_loss

    def forward_test(self, data):
        self.eval()
        with torch.no_grad():
            assert data.image.dim() == 4 and data.image.size(0) == 1 and data.image.size(1) == 3
            self.loc_layer.setGroundTruth(data.gt_boxes.to(self.device))
            H, W = data.image.size(2), data.image.size(3)
            self.loc_layer.setImageSize(H, W)
            self.features_ouput = self.net(data.image)
            empty = self.features_ouput.new()
            out_llm = self.llm(self.features_ouput, empty)
            captions = self.llm.decode_sequence(out_llm)
        return captions
