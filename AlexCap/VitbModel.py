import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from AlexCap.CustomLoss import CustomCrossEntropyLoss
from types import MethodType
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out, attention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention, alphas = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, alphas


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention, _ = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out, alphas = self.transformer_block(value, key, query, src_mask)
        return out, alphas


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.regul = torch.sqrt(torch.empty(1).fill_(embed_size))
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = heads

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x)*self.regul + self.position_embedding(positions)))
        for layer in self.layers:
            x, alphas = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)

        return out, alphas

class VitTransformer(nn.Module):
    def __init__(self, opt):

        super(VitTransformer, self).__init__()
        self.opt = opt
        if opt.trained_encoder:
            model_arch = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        else:
            model_arch = torchvision.models.vit_b_16()
        self.proj = model_arch.conv_proj
        self.class_token = model_arch.class_token
        self.encoder = model_arch.encoder
        if opt.trained_encoder:
            self.proj.requires_grad_(False)
            self.class_token.requires_grad_(False)
            self.encoder.requires_grad_(False)

        self.decoder = Decoder(
            opt.vocab_size+3,
            opt.embedding_size,
            opt.num_layers,
            8,
            4,
            opt.drop_value,
            opt.device,
            opt.seq_length+1,
        )
        self.sos = opt.vocab_size + 1
        self.eos = opt.vocab_size + 2
        self.device = opt.device
        self.token_dict = opt.idx_to_token
        self.max_length = opt.seq_length+1
        self.embed_size = opt.embedding_size
        self.use_beam = True
        self.beam_size = 4
        self.criterion = CustomCrossEntropyLoss()
        self.eval_mode = False
        self.llm = type('PrefixObject', (object,), {})()
        self.llm.decode_sequence = self.decode_sequence
        self.train()

    def set_eval(self, value):
        self.eval_mode = value

    def clip_gradient(self, norm):
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=norm)

    def make_trg_mask(self, trg, key_masking=True):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        if key_masking:
            lengths = ((trg > 0) * 1.0).unsqueeze(1)
            key_mask = torch.cat([torch.matmul(key.transpose(1, 0), key).unsqueeze(0) for key in lengths],
                                 dim=0).unsqueeze(1)
            trg_mask = trg_mask * key_mask
        return trg_mask.to(self.device)

    def decode_sequence(self, seq):
        if seq.dim() == 2:
            delimiter = ' '
            captions = []
            N, T = seq.size(0), seq.size(1)
            for i in range(N):
                caption = ''
                for t in range(T):
                    idx = seq[i, t]
                    if idx == self.eos or idx == 0:
                        # caption += '<EOS>'
                        break
                    if t > 0:
                        caption += delimiter
                    if idx == self.sos:
                        caption += '<SOS>'
                    else:
                        caption += self.token_dict[str(idx.item())]
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
                        if idx == self.eos or idx == 0:
                            break
                        if t > 0:
                            caption += delimiter
                        caption += self.token_dict[str(idx.item())]
                    captions.append(caption)
                captions_list.append(captions)
            return captions_list

    def get_target(self, gt_sequence, make_target=False):
        if make_target:
            N, T = gt_sequence.size(0), gt_sequence.size(1)
            target = torch.zeros(N, T + 1, dtype=gt_sequence.dtype)
            target[:, :T] = gt_sequence
            for i in range(N):
                for t in range(T + 1):
                    if target[i, t] == 0:
                        target[i, t] = self.eos
                        break
        else:
            N, T = gt_sequence.size(0), gt_sequence.size(1)
            target = torch.zeros(N, T + 1, dtype=gt_sequence.dtype)
            target[:, 0] = self.sos
            target[:, 1:T + 1] = gt_sequence
        return target

    def beam_search(self, input, beam_size):
        sentences = torch.empty(beam_size, 1).fill_(self.sos).long()
        input = input.expand(beam_size, input.size(1), input.size(2))
        top_preds = torch.zeros(beam_size, 1)

        completed_sentences = []
        completed_sentences_preds = []
        completed_sentences_alphas = []

        step = 1
        while True:
            trg_mask = self.make_trg_mask(sentences, False)
            output, alphas = self.decoder(sentences, input, None, trg_mask)
            output = output[:, -1]
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)

            prev_words_idxs = top_words // output.size(1)
            next_words_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_words_idxs], next_words_idxs.unsqueeze(1)), dim=1)
            incomplete = [idx for idx, next_word in enumerate(next_words_idxs) if next_word != self.eos]
            complete = list(set(range(len(next_words_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete])
                completed_sentences_preds.extend(top_preds[complete])
                completed_sentences_alphas.extend(alphas[complete].mean(dim=1)[:, :-1, 1:])

            beam_size -= len(complete)

            if beam_size == 0:
                break

            sentences = sentences[incomplete]
            input = input[:beam_size]
            top_preds = top_preds[incomplete].unsqueeze(1)

            if step >= self.max_length:
                break
            step += 1

        if len(completed_sentences_preds) == 0:
            sentence = sentences[0][1:].unsqueeze(0)
            alphas = alphas[0].mean(dim=0)[:, 1:]
        else:
            idx = completed_sentences_preds.index(max(completed_sentences_preds))
            sentence = completed_sentences[idx][1:].unsqueeze(0)
            alphas = completed_sentences_alphas[idx]

        return sentence, alphas

    def forward_train(self, data):
        if self.eval_mode:
            with torch.no_grad():
                input = data.image
                trg = data.gt_labels
                x = self.proj(input)
                x = x.reshape(x.size(0), self.embed_size, -1)
                x = x.permute(0, 2, 1)
                batch_class_token = self.class_token.expand(x.size(0), -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = self.encoder(x)
                src_mask = None
                trg_mask = self.make_trg_mask(self.get_target(trg))
                out, alphas = self.decoder(self.get_target(trg), x, src_mask, trg_mask)
                gt_target = self.get_target(trg, True)
                loss = self.criterion(out, gt_target)

                return loss
        else:
            input = data.image
            trg = data.gt_labels
            x = self.proj(input)
            x = x.reshape(x.size(0), self.embed_size, -1)
            x = x.permute(0, 2, 1)
            batch_class_token = self.class_token.expand(x.size(0), -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.encoder(x)
            src_mask = None
            trg_mask = self.make_trg_mask(self.get_target(trg))
            out, alphas = self.decoder(self.get_target(trg), x, src_mask, trg_mask)
            gt_target = self.get_target(trg, True)
            loss = self.criterion(out, gt_target)
            loss.backward()

            return loss
    def forward_test(self, data):
        self.eval()
        with torch.no_grad():
            input = data.image
            x = self.proj(input)
            x = x.reshape(x.size(0), self.embed_size, -1)
            x = x.permute(0, 2, 1)
            batch_class_token = self.class_token.expand(x.size(0), -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.encoder(x)
            if self.use_beam:
                captions, alphas = self.beam_search(x, self.beam_size)
                return self.decode_sequence(captions), alphas
            else:
                generated_tokens = torch.zeros(input.size(0), self.max_length, dtype=torch.long)
                trg = torch.empty(x.size(0), 1, dtype=torch.long).fill_(self.sos)
                src_mask = None
                for i in range(self.max_length):
                    trg_mask = self.make_trg_mask(trg, key_masking=False)
                    out, alphas = self.decoder(trg, x, src_mask, trg_mask)
                    next_token = out[:, -1, :].argmax(dim=1, keepdim=True)
                    generated_tokens[:, i:i+1] = next_token
                    trg = torch.cat((trg, next_token), dim=1)
                    if torch.all(next_token == self.eos):
                        break
                captions = self.decode_sequence(generated_tokens)
                alphas = alphas.mean(dim=1).squeeze()[:-1, 1:]
                return captions, alphas