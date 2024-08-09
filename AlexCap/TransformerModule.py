import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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

        return out

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
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        patch_size
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.patch_length = patch_size*patch_size
        self.position_embedding = nn.Embedding(self.patch_length, embed_size)
        self.device = device
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        positions = torch.arange(0, self.patch_length).expand(x.size(0), self.patch_length).to(self.device)
        out = self.dropout(x + self.position_embedding(positions))
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


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
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


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

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x)*self.regul + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        fc_dim,
        token_dict,
        max_length,
        patch_size,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
    ):

        super(Transformer, self).__init__()

        self.fc = nn.Sequential(nn.Linear(fc_dim, embed_size), nn.ReLU(inplace=True))
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            patch_size
        )

        self.decoder = Decoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.sos = src_vocab_size -2
        self.eos = src_vocab_size -1
        self.device = device
        self.token_dict = token_dict
        self.max_length = max_length
        self.use_beam = False
        self.beam_size = 3

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

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

    def beam_search(self, input, beam_width):
        lsm = torch.nn.LogSoftmax(-1)
        with torch.no_grad():
            generated_tokens = torch.zeros(beam_width, self.max_length)
            word_vec = self.fc(input).unsqueeze(1)
            trg = torch.ones(word_vec.size(0), 1, dtype=torch.long) * self.sos
            src_mask = None
            enc_src = self.encoder(word_vec, src_mask)
            trg_mask = self.make_trg_mask(trg)
            out = self.decoder(trg, enc_src, src_mask, trg_mask)
            prob, top_idx = torch.topk(lsm(out[:, -1, :]), k=beam_width, dim=-1)
            top_idx = top_idx.reshape(-1, 1)
            next_beams = trg.expand(beam_width, 1)
            next_beams = torch.cat((next_beams, top_idx), dim=1)
            generated_tokens[:, 0:1] = top_idx
            enc_src = enc_src.expand(beam_width, -1, -1)
            voc_size = out.size(-1)
            for i in range(1, self.max_length):
                beam_mask = self.make_trg_mask(next_beams)
                out = self.decoder(next_beams, enc_src, src_mask, beam_mask)
                next_prob = lsm(out[:, -1, :])
                end_mask = torch.eq(top_idx[:, 0], self.eos)
                next_prob[end_mask, :self.eos] = -100
                next_prob[end_mask, self.eos] = 0
                prob = next_prob + prob.reshape(-1, 1)
                prob = torch.flatten(prob, start_dim=0)
                prob, idx = torch.topk(prob, k=beam_width, dim=-1)
                top_idx = torch.remainder(idx, voc_size).unsqueeze(-1)
                best_candidates = (idx / voc_size).long()

                generated_tokens = generated_tokens[best_candidates, :]
                generated_tokens[:, i: i+1] = top_idx
                if torch.all(top_idx == self.eos):
                    break
                next_beams = next_beams[best_candidates, :]
                next_beams = torch.cat((next_beams, top_idx), dim=1)
        return generated_tokens

    def forward(self, input, trg):
        if trg.nelement() > 0:
            word_vec = self.fc(input)
            src_mask = None
            trg_mask = self.make_trg_mask(self.get_target(trg))
            enc_src = self.encoder(word_vec, src_mask)
            out = self.decoder(self.get_target(trg), enc_src, src_mask, trg_mask)

            return out
        else:
            if self.use_beam:
                res = torch.zeros(input.size(0), self.beam_size, self.max_length)
                for i in range(input.size(0)):
                    res[i] = self.beam_search(input[i:i+1], self.beam_size)
                return res.int()
                # return self.beam_search(input, 3)
            else:
                with torch.no_grad():
                    generated_tokens = torch.zeros(input.size(0), self.max_length, dtype=torch.long)
                    word_vec = self.fc(input)
                    trg = torch.empty(word_vec.size(0), 1, dtype=torch.long).fill_(self.sos)
                    src_mask = None
                    enc_src = self.encoder(word_vec, src_mask)

                    for i in range(self.max_length):
                        trg_mask = self.make_trg_mask(trg)
                        out = self.decoder(trg, enc_src, src_mask, trg_mask)
                        next_token = out[:, -1, :].argmax(dim=1, keepdim=True)
                        generated_tokens[:, i:i+1] = next_token
                        trg = torch.cat((trg, next_token), dim=1)
                        if torch.all(next_token == self.eos):
                            break

                return generated_tokens

class Transformer2(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        fc_dim,
        token_dict,
        max_length,
        patch_size,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
    ):

        super(Transformer2, self).__init__()
        self.trans = nn.Transformer(embed_size, heads, num_layers, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(fc_dim, embed_size), nn.ReLU(inplace=True))
        self.fc_out = nn.Linear(embed_size, src_vocab_size)
        self.embedding = nn.Embedding(src_vocab_size, embed_size)
        self.input_encoding = nn.Embedding(patch_size*patch_size, embed_size)
        self.target_encoding = nn.Embedding(max_length, embed_size)
        self.d_model = torch.empty(1, device=device).fill_(embed_size)
        self.sos = src_vocab_size - 2
        self.eos = src_vocab_size - 1
        self.device = device
        self.token_dict = token_dict
        self.max_length = max_length
        self.temperature = 0.1

    def generate_subsequent_masks(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

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

    def beam_search(self, input, beam_width):
        lsm = torch.nn.LogSoftmax(-1)
        with torch.no_grad():
            generated_tokens = torch.zeros(beam_width, self.max_length)
            word_vec = self.fc(input).unsqueeze(1)
            trg = torch.ones(word_vec.size(0), 1, dtype=torch.long) * self.sos
            src_mask = None
            enc_src = self.encoder(word_vec, src_mask)
            trg_mask = self.make_trg_mask(trg)
            out = self.decoder(trg, enc_src, src_mask, trg_mask)
            prob, top_idx = torch.topk(lsm(out[:, -1, :]), k=beam_width, dim=-1)
            top_idx = top_idx.reshape(-1, 1)
            next_beams = trg.expand(beam_width, 1)
            next_beams = torch.cat((next_beams, top_idx), dim=1)
            generated_tokens[:, 0:1] = top_idx
            enc_src = enc_src.expand(beam_width, -1, -1)
            voc_size = out.size(-1)
            for i in range(1, self.max_length):
                beam_mask = self.make_trg_mask(next_beams)
                out = self.decoder(next_beams, enc_src, src_mask, beam_mask)
                next_prob = lsm(out[:, -1, :])
                end_mask = torch.eq(top_idx[:, 0], self.eos)
                next_prob[end_mask, :self.eos] = -100
                next_prob[end_mask, self.eos] = 0
                prob = next_prob + prob.reshape(-1, 1)
                prob = torch.flatten(prob, start_dim=0)
                prob, idx = torch.topk(prob, k=beam_width, dim=-1)
                top_idx = torch.remainder(idx, voc_size).unsqueeze(-1)
                best_candidates = (idx / voc_size).long()

                generated_tokens = generated_tokens[best_candidates, :]
                generated_tokens[:, i: i+1] = top_idx
                if torch.all(top_idx == self.eos):
                    break
                next_beams = next_beams[best_candidates, :]
                next_beams = torch.cat((next_beams, top_idx), dim=1)
        return generated_tokens

    def forward(self, input, sequences):
        if sequences.nelement() > 0:
            input_positions = torch.arange(0, input.size(1)).expand(input.size(0), input.size(1)).to(self.device)
            encoder_input = self.fc(input) + self.input_encoding(input_positions)

            target = self.get_target(sequences)
            target_positions = torch.arange(0, target.size(1)).expand(target.size(0), target.size(1)).to(self.device)
            embed_target = self.embedding(target) * torch.sqrt(self.d_model) + self.target_encoding(target_positions)

            key_mask = target == 0
            trg_mask = (self.trans.generate_square_subsequent_mask(target.size(1)) < 0).to(self.device)

            decoder_output = self.trans(encoder_input, embed_target, tgt_mask=trg_mask, tgt_key_padding_mask=key_mask)
            output = self.fc_out(decoder_output)
            return output #F.log_softmax(output, dim=-1)

        else:
            generated_tokens = torch.zeros(input.size(0), self.max_length, dtype=torch.long)
            input_positions = torch.arange(0, input.size(1)).expand(input.size(0), input.size(1)).to(self.device)
            encoder_input = self.fc(input) + self.input_encoding(input_positions)
            trg = torch.empty(input.size(0), 1, dtype=torch.long).fill_(self.sos)

            for i in range(self.max_length):
                trg_mask = (self.trans.generate_square_subsequent_mask(trg.size(1)) < 0).to(self.device)
                target_positions = torch.arange(0, trg.size(1)).expand(trg.size(0), trg.size(1)).to(self.device)
                embed_trg = self.embedding(trg) + self.target_encoding(target_positions)
                decoder_output = self.trans(encoder_input, embed_trg, tgt_mask=trg_mask)
                weights = torch.exp(F.log_softmax(self.fc_out(decoder_output)[:, -1], dim=-1) / self.temperature)
                next_token = torch.multinomial(weights, 1)
                generated_tokens[:, i:i + 1] = next_token
                trg = torch.cat((trg, next_token), dim=1)
                if torch.all(next_token == self.eos):
                    break
            # generated_tokens = torch.zeros(input.size(0), self.max_length, dtype=torch.long)
            # encoder_input = self.fc(input)
            # trg = torch.empty(input.size(0), 1, dtype=torch.long).fill_(self.sos)
            #
            # for i in range(self.max_length):
            #     trg_mask = (self.trans.generate_square_subsequent_mask(trg.size(1)) < 0).to(self.device)
            #     embed_trg = self.embedding(trg)
            #     decoder_output = self.trans(encoder_input, embed_trg, tgt_mask=trg_mask)
            #     out = self.fc_out(decoder_output)
            #     next_token = out[:, -1, :].argmax(dim=1, keepdim=True)
            #     generated_tokens[:, i:i + 1] = next_token
            #     trg = torch.cat((trg, next_token), dim=1)
            #     if torch.all(next_token == self.eos):
            #         break

            return generated_tokens

