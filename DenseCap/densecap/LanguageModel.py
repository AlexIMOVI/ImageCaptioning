import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, Softmax, LogSoftmax
from torch.nn.functional import dropout

from densecap_utils import utils

class ViewLayer(nn.Module):
    def __init__(self,dims):
        super(ViewLayer,self).__init__()
        self.dims = dims
    def forward(self, inputs):
        if len(self.dims) == 3:
            return inputs.contiguous().view(self.dims[0], self.dims[1], self.dims[2])
        else:
            return inputs.contiguous().view(self.dims[0], self.dims[1])
class LanguageModel(nn.Module):
    def __init__(self, opt=None):
        super(LanguageModel, self).__init__()

        opt = opt or {}
        self.vocab_size = utils.getopt(opt, 'vocab_size')
        self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
        self.image_vector_dim = utils.getopt(opt, 'image_vector_dim')
        self.rnn_size = utils.getopt(opt, 'rnn_size')
        self.seq_length = utils.getopt(opt, 'seq_length')
        self.num_layers = utils.getopt(opt, 'num_layers', 1)
        self.idx_to_token = utils.getopt(opt, 'idx_to_token')
        self.dropout = utils.getopt(opt, 'dropout', 0)
        self.device = utils.getopt(opt,'device')
        W, D = self.input_encoding_size, self.image_vector_dim
        V, H = self.vocab_size, self.rnn_size

        self.image_encoder = nn.Sequential()
        self.image_encoder.add_module('encode', nn.Linear(D, W))
        self.image_encoder.add_module('relu', nn.ReLU(inplace=True))

        self.START_TOKEN = self.vocab_size + 1
        self.END_TOKEN = self.vocab_size + 2
        self.NULL_TOKEN = 0

        V, W = self.vocab_size, self.input_encoding_size
        self.lookup_table = nn.Embedding(V + 3, W, device=self.device)

        self.sample_argmax = True

        input_dim = self.input_encoding_size

        self.rnn = nn.Sequential()

        self.lstm = LSTM(input_dim, self.rnn_size, self.num_layers, dropout=self.dropout, batch_first=True, device=self.device)
        self.view_in = ViewLayer([1, 1, -1])
        self.view_out = ViewLayer([1, -1])
        self.rnn.add_module('drop', nn.Dropout(0.5))
        self.rnn.add_module('linear', Linear(H, V + 3))


        self._forward_sampled = False
        self.recompute_backward = True

        self.train()

    def decode_sequence(self, seq):
        delimiter = ' '
        captions = []
        N, T = seq.size(0), seq.size(1)
        for i in range(N):
            caption = ''
            for t in range(T):
                idx = seq[i, t]
                if idx == self.END_TOKEN or idx == 0:
                    caption += '<EOS>'
                    break
                if t > 0:
                    caption += delimiter
                if idx == self.START_TOKEN:
                    caption += '<SOS>'
                else:
                    caption += self.idx_to_token[str(idx.item())]
            captions.append(caption)
        return captions

    def forward(self, input):

        image_vectors, gt_sequence = input[0], input[1]

        if gt_sequence.nelement() > 0:
            N, T = gt_sequence.size(0), gt_sequence.size(1)
            self._gt_with_start = torch.zeros(N, T + 1, dtype=torch.long)
            self._gt_with_start[:, 0] = self.START_TOKEN
            self._gt_with_start[:, 1:T + 1] = gt_sequence

            self.view_in.dims = [N * (T + 2), -1]
            self.view_out.dims = [N, T + 2, -1]

            word_vectors = self.image_encoder(image_vectors).unsqueeze(1)
            word_vectors_gt = self.lookup_table(self._gt_with_start)
            lstm_out, state = self.lstm.forward(torch.concat((word_vectors,word_vectors_gt), dim=1))
            self.output = self.rnn.forward(lstm_out)
            self._forward_sampled = False
            return self.output
        else:
            self._forward_sampled = True
            return self.predict_caption(image_vectors)

    def get_target(self, gt_sequence, make_target=False):
        if make_target:
            N, T = gt_sequence.size(0), gt_sequence.size(1)
            target = torch.zeros(N, T + 2, dtype=gt_sequence.dtype)
            target[:, 0] = self.START_TOKEN
            target[:, 1:T + 1] = gt_sequence
            for i in range(N):
                for t in range(1, T + 2):
                    if target[i, t] == 0:
                        target[i, t] = self.END_TOKEN
                        break
        else:
            N, T = gt_sequence.size(0), gt_sequence.size(1)
            target = torch.zeros(N, T + 2, dtype=gt_sequence.dtype)
            target[:, 0] = 1
            target[:, 1] = self.START_TOKEN
            target[:, 2:T + 2] = gt_sequence
        return target

    def predict_caption(self, image_vectors):
        with torch.no_grad():
            word_vectors = self.image_encoder(image_vectors).unsqueeze(1)
            seq_list = torch.zeros(word_vectors.size(0), self.seq_length, dtype=torch.long, device=self.device)
            lsm = nn.LogSoftmax(dim=-1)
            T = self.seq_length
            hiddens, states = self.lstm(word_vectors)
            start_vec = torch.ones(word_vectors.size(0), 1, dtype=torch.long) * self.START_TOKEN
            input_vec = self.lookup_table(start_vec)
            for i in range(T):
                hiddens, states = self.lstm(input_vec, hiddens)
                output = self.rnn.forward(hiddens)
                score = lsm.forward(output)
                best_score = score[:, -1, :].argmax(dim=1)
                input_vec = self.lookup_table(best_score)
                if torch.all(best_score == self.END_TOKEN):
                    seq_list[:, i] = self.END_TOKEN
                    break
                seq_list[:, i:i+1] = best_score

        return seq_list

    def parameters(self):
        return self.net.parameters()

    def training(self):
        self.net.train()

    def evaluate(self):
        self.net.eval()

    def clearState(self):
        self.net.clearState()
