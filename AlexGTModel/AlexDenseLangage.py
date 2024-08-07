import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, Softmax, LogSoftmax
import torch.nn.functional as F

class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        input_encoding_size,
        image_vector_dim,
        rnn_size,
        seq_length,
        num_layers,
        idx_to_token,
        dropout,
        device,
        curriculum_learning
    ):

        super(LanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self.image_vector_dim = image_vector_dim
        self.rnn_size = rnn_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.idx_to_token = idx_to_token
        self.dropout = dropout
        self.device = device

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
        input_dim = self.input_encoding_size
        self.use_beam = False
        self.beam_size = 3
        self.use_curriculum = curriculum_learning
        self.teacher_prob = 1
        self.rnn = nn.Sequential()

        self.lstm = LSTM(input_dim, self.rnn_size, self.num_layers, batch_first=True, dropout=self.dropout, device=self.device)

        # self.rnn.add_module('drop', nn.Dropout(self.dropout))
        self.rnn.add_module('linear', Linear(H, V + 3))
        self.train()

    def decode_sequence(self, seq):
        if seq.dim() == 2:
            delimiter = ' '
            captions = []
            N, T = seq.size(0), seq.size(1)
            for i in range(N):
                caption = ''
                for t in range(T):
                    idx = seq[i, t]
                    if idx == self.END_TOKEN or idx == 0:
                        break
                    if t > 0:
                        caption += delimiter
                    caption += self.idx_to_token[str(idx.item())]
                captions.append(caption)
            return captions
        else:
            captions_list = []
            N, T = seq.size(1), seq.size(2)
            delimiter = ' '
            for s in seq:
                captions = []
                for i in range(N):
                    caption = ''
                    for t in range(T):
                        idx = s[i, t]
                        if idx == self.END_TOKEN or idx == 0:
                            break
                        if t > 0:
                            caption += delimiter
                        caption += self.idx_to_token[str(idx.item())]
                    captions.append(caption)
                captions_list.append(captions)
            return captions_list

    def forward(self, image_vectors, gt_sequence):
        if gt_sequence.nelement() > 0:
            if self.use_curriculum:
                return self.teacher_learning(image_vectors, gt_sequence)
            else:
                self._gt_with_start = self.get_target(gt_sequence)
                word_vectors = self.image_encoder(image_vectors).unsqueeze(1)
                word_vectors_gt = self.lookup_table(self._gt_with_start)
                lstm_out, state = self.lstm(torch.concat((word_vectors, word_vectors_gt), dim=1))
                self.output = self.rnn(lstm_out)[:, 1:, :]
                return self.output.contiguous()
        else:
            if self.use_beam:
                return self.beam_search(image_vectors, self.beam_size).int()
            return self.predict_caption(image_vectors)

    def get_target(self, gt_sequence, make_target=False):
        if make_target:
            N, T = gt_sequence.size(0), gt_sequence.size(1)
            target = torch.zeros(N, T + 1, dtype=gt_sequence.dtype)
            target[:, 0:T] = gt_sequence
            for i in range(N):
                for t in range(1, T + 1):
                    if target[i, t] == 0:
                        target[i, t] = self.END_TOKEN
                        break
        else:
            N, T = gt_sequence.size(0), gt_sequence.size(1)
            target = torch.zeros(N, T + 1, dtype=gt_sequence.dtype)
            target[:, 0] = self.START_TOKEN
            target[:, 1: T+1] = gt_sequence
        return target

    def predict_caption(self, image_vectors):
        with torch.no_grad():
            word_vectors = self.image_encoder(image_vectors).unsqueeze(1)
            seq_list = torch.zeros(word_vectors.size(0), self.seq_length+1, dtype=torch.long, device=self.device)
            lsm = nn.LogSoftmax(dim=-1)
            _, states = self.lstm(word_vectors)
            start_vec = torch.ones(word_vectors.size(0), 1, dtype=torch.long) * self.START_TOKEN
            input_vec = self.lookup_table(start_vec)
            for i in range(self.seq_length+1):
                hiddens, states = self.lstm(input_vec, states)
                output = self.rnn(hiddens)
                score = lsm(output)
                best_score = score[:, -1, :].argmax(dim=1)
                input_vec = self.lookup_table(best_score).unsqueeze(1)

                if torch.all(best_score == self.END_TOKEN):
                    seq_list[:, i] = self.END_TOKEN
                    break
                seq_list[:, i] = best_score

        return seq_list

    def teacher_learning(self, image_vectors, gt_sequences):
        word_vectors = self.image_encoder(image_vectors).unsqueeze(1)
        seq_list = torch.zeros(word_vectors.size(0), self.seq_length+1, self.vocab_size+3, device=self.device)
        lsm = nn.LogSoftmax(dim=-1)
        _, states = self.lstm(word_vectors)
        gt_vec = self.lookup_table(gt_sequences)
        input_vec = gt_vec[:, 0:1, :]
        for i in range(self.seq_length+1):
            hiddens, states = self.lstm(input_vec, states)
            output = self.rnn(hiddens)
            if i < self.seq_length:
                prob = torch.rand(1)
                if prob > self.teacher_prob:
                    score = lsm(output)
                    best_score = score[:, -1, :].argmax(dim=1)
                    input_vec = self.lookup_table(best_score).unsqueeze(1)
                else:
                    input_vec = gt_vec[:, i+1:i+2, :]
            seq_list[:, i] = output[:, -1, :]

        return seq_list

    def beam_search(self, image_vectors, beam_size):
        with torch.no_grad():
            batch_size = image_vectors.size(0)
            word_beam_vectors = self.image_encoder(image_vectors).unsqueeze(1).unsqueeze(0)
            word_beam_vectors = word_beam_vectors.expand(beam_size, word_beam_vectors.size(1), word_beam_vectors.size(2), word_beam_vectors.size(3))
            word_beam_vectors = word_beam_vectors.permute(1, 0, 2, 3).flatten(end_dim=1)
            generated_tokens = torch.zeros(word_beam_vectors.size(0), self.seq_length)
            lsm = nn.LogSoftmax(dim=-1)
            _, states = self.lstm(word_beam_vectors)
            start_vec = torch.ones(word_beam_vectors.size(0), 1, dtype=torch.long) * self.START_TOKEN
            input_vec = self.lookup_table(start_vec)
            hiddens, states = self.lstm(input_vec, states)
            output = self.rnn(hiddens)
            prob, top_idx = torch.topk(lsm(output[::beam_size, -1, :]), k=beam_size, dim=-1)
            top_idx = top_idx.flatten().unsqueeze(-1)
            next_beams = self.lookup_table(top_idx)
            generated_tokens[:, 0:1] = top_idx
            voc_size = output.size(2)
            lvl = torch.arange(batch_size) * beam_size
            lvl = lvl.unsqueeze(-1)
            for i in range(1, self.seq_length):
                hiddens, states = self.lstm(next_beams, states)
                output = self.rnn(hiddens)
                next_prob = lsm(output[:, -1, :])
                end_mask = torch.eq(top_idx[:, 0], self.END_TOKEN)
                next_prob[end_mask, :self.END_TOKEN] = -100
                next_prob[end_mask, self.END_TOKEN] = 0
                prob = next_prob + prob.flatten().unsqueeze(-1)
                prob = prob.reshape(batch_size, beam_size*voc_size)
                prob, idx = torch.topk(prob, k=beam_size, dim=-1)
                top_idx = torch.remainder(idx, voc_size).flatten().unsqueeze(-1)
                best_candidates = (idx / voc_size).long() + lvl
                best_candidates = best_candidates.flatten()
                generated_tokens = generated_tokens[best_candidates, :]
                generated_tokens[:, i:i+1] = top_idx
                if torch.all(top_idx == self.END_TOKEN):
                    break
                next_beams = self.lookup_table(top_idx)

            return generated_tokens.reshape(batch_size, beam_size, self.seq_length)


