import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, Softmax, LogSoftmax
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, rnn_size):
        super(Attention, self).__init__()
        self.U = nn.Linear(rnn_size, rnn_size)
        self.W = nn.Linear(encoder_dim, rnn_size)
        self.v = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha
class Decoder(nn.Module):
    def __init__(self,
                vocab_size,
                embedding_size,
                image_vector_dim,
                rnn_size,
                seq_length,
                idx_to_token,
                dropout,
                device,
        ):
        super(Decoder, self).__init__()
        self.vocabulary_size = vocab_size
        self.embedding_size = embedding_size
        self.image_vector_dim = image_vector_dim
        self.rnn_size = rnn_size
        self.seq_length = seq_length
        self.idx_to_token = idx_to_token
        self.dropout = dropout
        self.device = device
        self.START_TOKEN = self.vocabulary_size + 1
        self.END_TOKEN = self.vocabulary_size + 2

        self.init_h = nn.Linear(image_vector_dim, rnn_size)
        self.init_c = nn.Linear(image_vector_dim, rnn_size)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(rnn_size, image_vector_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(rnn_size, vocab_size+3)
        self.dropout = nn.Dropout()

        self.attention = Attention(image_vector_dim, rnn_size)
        self.embedding = nn.Embedding(vocab_size+3, embedding_size)
        self.lstm = nn.LSTMCell(embedding_size + image_vector_dim, rnn_size)
        self.use_beam = False
        self.beam_size = 3

    def forward(self, img_features, captions):
        if captions.nelement() > 0:
            batch_size = img_features.size(0)

            h, c = self.get_init_lstm_state(img_features)
            max_timespan = self.seq_length + 1
            captions = self.get_target(captions)
            embedding = self.embedding(captions)
            preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size+3).cuda()
            alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).cuda()
            for t in range(max_timespan):
                context, alpha = self.attention(img_features, h)
                gate = self.sigmoid(self.f_beta(h))
                gated_context = gate * context
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)

                h, c = self.lstm(lstm_input, (h, c))
                output = self.deep_output(self.dropout(h))

                preds[:, t] = output
                alphas[:, t] = alpha

            return preds, alphas
        else:
            if self.use_beam:
                return self.caption(img_features, self.beam_size)
            else:
                batch_size = img_features.size(0)

                h, c = self.get_init_lstm_state(img_features)
                max_timespan = self.seq_length + 1
                prev_words = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.START_TOKEN)
                embedding = self.embedding(prev_words)

                preds = torch.zeros(batch_size, self.seq_length+1, dtype=torch.long, device=self.device)
                alphas = torch.zeros(batch_size, max_timespan, img_features.size(1))
                for t in range(max_timespan):
                    context, alpha = self.attention(img_features, h)
                    gate = self.sigmoid(self.f_beta(h))
                    gated_context = gate * context

                    embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                    lstm_input = torch.cat((embedding, gated_context), dim=1)

                    h, c = self.lstm(lstm_input, (h, c))
                    output = self.deep_output(self.dropout(h))
                    token = output.argmax(1).reshape(batch_size, 1)
                    preds[:, t] = token
                    alphas[:, t] = alpha
                    if torch.all(token == self.END_TOKEN):
                        break
                    embedding = self.embedding(token)

                return preds, alphas
    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c


    def caption(self, img_features, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        prev_words = torch.empty(beam_size, 1).fill_(self.START_TOKEN).long()
        img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words // output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != self.END_TOKEN]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete])
                completed_sentences_alphas.extend(alphas[complete])
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > self.seq_length:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx][1:].unsqueeze(0)
        alpha = completed_sentences_alphas[idx][1:].unsqueeze(0)

        return sentence, alpha

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
                if idx == self.START_TOKEN:
                    caption += 'SOS'
                else:
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
                        if idx == self.START_TOKEN:
                            caption += 'SOS'
                        else:
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
                            if idx == self.START_TOKEN:
                                caption += 'SOS'
                            else:
                                caption += self.idx_to_token[str(idx.item())]
                        captions.append(caption)
                    captions_list.append(captions)
                return captions_list