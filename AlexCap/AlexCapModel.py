import torchvision
import torch
import torch.nn as nn
from AlexCap.AlexTransformer import Transformer2
from AlexCap.LSTMLoss import CustomCrossEntropyLoss, CustomNLLLoss
from AlexCap.LanguageModel import LanguageModel as LLM
from AlexCap.AttentionLanguageModel import Decoder as LLM2

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
        if opt.use_vggface:
            backbone_model = torchvision.models.vgg16()
            face_weights = 'AlexCap/models_pth/pytorch_vggface_weights.pth'
            backbone_model.load_state_dict(torch.load(face_weights))
            self.features = nn.Sequential(*list(backbone_model.features.children())[:-1])
            fc_dim = 512
        else:
            resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            fc_dim = 2048

        if opt.use_lstm:
            if opt.use_attention:
                self.llm = LLM2(opt.vocab_size,
                                 opt.embedding_size,
                                 fc_dim,
                                 opt.lstm_size,
                                 opt.seq_length,
                                 opt.num_layers,
                                 opt.idx_to_token,
                                 self.dropout,
                                 self.device)
            else:
                self.llm = LLM(opt.vocab_size,
                                 opt.embedding_size,
                                 fc_dim,
                                 opt.lstm_size,
                                 opt.seq_length,
                                 opt.num_layers,
                                 opt.idx_to_token,
                                 self.dropout,
                                 self.device,
                                 opt.use_curriculum_learning)
        else:
            patch_size = 14 if opt.use_vggface else 7
            self.llm = Transformer2(src_vocab_size=opt.vocab_size + 3,
                                   fc_dim=fc_dim,
                                   token_dict=opt.idx_to_token,
                                   max_length=opt.seq_length+1,
                                   patch_size=patch_size,
                                   embed_size=opt.embedding_size,
                                   dropout=self.dropout,
                                   device=self.device)

        # criteria
        if opt.use_lstm:
            self.crit = CustomCrossEntropyLoss()
        else:
            self.crit = CustomCrossEntropyLoss()#CustomNLLLoss()
        self.llm.use_curriculum = opt.use_curriculum_learning
        self.convergence_rate = torch.tensor([5000], device=self.device)
        self.eval_mode = False
        self.train()

    def set_eval(self, value):
        self.eval_mode = value

    def set_teacher_prob(self, iter):
        self.llm.teacher_prob = self.convergence_rate / (2*self.convergence_rate + torch.exp(iter / (2*self.convergence_rate))) + 0.5

    def clip_gradient(self, norm):
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=norm)

    def forward_train(self, data):
        gt_labels = data.gt_labels
        if self.eval_mode:
            with torch.no_grad():
                assert data.image.dim() == 4 and data.image.size(1) == 3
                features_ouput = self.features(data.image)
                features_ouput = features_ouput.permute(0, 2, 3, 1)
                features_ouput = features_ouput.view(features_ouput.size(0), -1, features_ouput.size(-1))
                out_llm = self.llm.forward(features_ouput, gt_labels)
                gt_targets = self.llm.get_target(gt_labels, make_target=True)
                if self.opt.use_attention:
                    preds, alphas = out_llm
                    captioning_loss = self.crit.forward(preds, gt_targets)
                    att_regul = ((1 - alphas.sum(1)) ** 2).mean()
                    captioning_loss += att_regul
                else:
                    captioning_loss = self.crit.forward(out_llm, gt_targets)

        else:
            assert data.image.dim() == 4 and data.image.size(1) == 3
            features_ouput = self.features(data.image)
            features_ouput = features_ouput.permute(0, 2, 3, 1)
            features_ouput = features_ouput.view(features_ouput.size(0), -1, features_ouput.size(-1))
            out_llm = self.llm.forward(features_ouput, gt_labels)
            gt_targets = self.llm.get_target(gt_labels, make_target=True)
            if self.opt.use_attention:
                preds, alphas = out_llm
                captioning_loss = self.crit.forward(preds, gt_targets)
                att_regul = ((1 - alphas.sum(1)) ** 2).mean()
                captioning_loss += att_regul
            else:
                captioning_loss = self.crit.forward(out_llm, gt_targets)
            captioning_loss.backward()
        return captioning_loss

    def forward_test(self, data):
        self.eval()
        with torch.no_grad():
            # assert data.image.dim() == 4 and data.image.size(0) == 1 and data.image.size(1) == 3
            features_ouput = self.features(data.image)
            features_ouput = features_ouput.permute(0, 2, 3, 1)
            features_ouput = features_ouput.view(features_ouput.size(0), -1, features_ouput.size(-1))
            empty = features_ouput.new()
            out_llm = self.llm.forward(features_ouput, empty)
            if self.opt.use_attention:
                preds, alphas = out_llm
                captions = self.llm.decode_sequence(preds)
                return captions, alphas
            else:
                captions = self.llm.decode_sequence(out_llm)
        return captions


