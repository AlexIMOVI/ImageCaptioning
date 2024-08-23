import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, Softmax, LogSoftmax
import torch.nn.functional as F

class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        image_vector_dim,
        rnn_size,
        seq_length,
        num_layers,
        idx_to_token,
        dropout,
        device
    ):

        super(LanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.image_vector_dim = image_vector_dim
        self.rnn_size = rnn_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.idx_to_token = idx_to_token
        self.dropout = dropout
        self.device = device

        W, D = self.embedding_size, self.image_vector_dim
        V, H = self.vocab_size, self.rnn_size

        self.image_encoder = nn.Sequential()
        self.image_encoder.add_module('encode', nn.Linear(D, W))
        self.image_encoder.add_module('relu', nn.ReLU(inplace=True))

        self.START_TOKEN = self.vocab_size + 1
        self.END_TOKEN = self.vocab_size + 2
        self.NULL_TOKEN = 0
        self.lookup_table = nn.Embedding(V + 3, W, device=self.device)
        self.use_beam = False
        self.beam_size = 3
        self.rnn = nn.Sequential()

        self.lstm = LSTM(W, self.rnn_size, self.num_layers, batch_first=True, device=self.device)

        self.rnn.add_module('drop', nn.Dropout(self.dropout))
        self.rnn.add_module('linear', Linear(H, V + 3))
        self.train()

    def decode_sequence(self, seq):
        if isinstance(seq, list):
            caption = ''
            delimiter = ' '
            for t in range(1, len(seq)):
                idx = seq[t]
                if idx == self.END_TOKEN or idx == 0:
                    break
                if t > 1:
                    caption += delimiter
                caption += self.idx_to_token[str(idx)]
            return [caption]
        else:
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
            self._gt_with_start = self.get_target(gt_sequence)
            word_vectors_gt = self.lookup_table(self._gt_with_start)
            encoded_vectors = self.image_encoder(image_vectors)
            _, state = self.lstm(encoded_vectors)
            lstm_out, state = self.lstm(word_vectors_gt, state)
            self.output = self.rnn(lstm_out)
            return self.output.contiguous()
        else:
            if self.use_beam:
                return self.caption(image_vectors, self.beam_size)
            return self.predict_caption(image_vectors)

    def get_target(self, gt_sequence, make_target=False):
        if make_target:
            N, T = gt_sequence.size(0), gt_sequence.size(1)
            target = torch.zeros(N, T + 1, dtype=gt_sequence.dtype)
            target[:, :T] = gt_sequence
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
            image_vectors = self.image_encoder(image_vectors)
            seq_list = torch.zeros(image_vectors.size(0), self.seq_length+1, dtype=torch.long, device=self.device)
            lsm = nn.LogSoftmax(dim=-1)
            _, states = self.lstm(image_vectors)
            start_vec = torch.ones(image_vectors.size(0), 1, dtype=torch.long) * self.START_TOKEN
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

    def caption(self, img_features, beam_size):
        with torch.no_grad():
            prev_words = torch.empty(beam_size, 1).long().fill_(self.START_TOKEN)
            sentences = prev_words
            top_preds = torch.zeros(beam_size, 1)

            completed_sentences = []
            completed_sentences_preds = []
            _, states = self.lstm(img_features.expand(beam_size, img_features.size(1), img_features.size(2)))
            step = 1
            while True:
                embedding = self.lookup_table(prev_words)
                output, states = self.lstm(embedding, states)
                output = self.rnn(output)
                output = top_preds.unsqueeze(2).expand_as(output) + output

                if step == 1:
                    top_preds, top_words = output[0, 0].topk(beam_size, 0)
                else:
                    top_preds, top_words = output.view(-1).topk(beam_size, 0)
                prev_word_idxs = top_words // output.size(2)
                next_word_idxs = top_words % output.size(2)

                sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)

                incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != self.END_TOKEN]
                complete = list(set(range(len(next_word_idxs))) - set(incomplete))

                if len(complete) > 0:
                    completed_sentences.extend(sentences[complete].tolist())
                    completed_sentences_preds.extend(top_preds[complete])
                beam_size -= len(complete)

                if beam_size == 0:
                    break
                sentences = sentences[incomplete]
                states = (states[0][:, prev_word_idxs[incomplete]], states[1][:, prev_word_idxs[incomplete]])
                top_preds = top_preds[incomplete].unsqueeze(1)
                prev_words = next_word_idxs[incomplete].unsqueeze(1)

                if step > self.seq_length+1:
                    break
                step += 1

            idx = completed_sentences_preds.index(max(completed_sentences_preds))
            sentence = completed_sentences[idx]
        return sentence
