import torch
import torch.nn as nn
import numpy as np

"""
Module for NLI
"""


class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # encoder
        self.encoder = BiLSTMMaxPoolEncoder(config)
        self.lstm_dim = config['lstm_dim']
        self.input_dim = 4 * 2 * self.lstm_dim

        # classifier
        self.mlp_dim = config['mlp_dim']
        self.output_dim = config['output_dim']
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_dim, self.output_dim)
        )

    def forward(self, premises, hypothesises):
        # premise : (premises_batch, premises_len)
        u = self.encoder(premises)
        v = self.encoder(hypothesises)
        # concat [u, v, |u-v|, u * v]
        features = torch.cat([u, v, torch.abs(u-v), u*v], 1)
        output = self.classifier(features)
        return output


"""
Bidirectional LSTM with max pooling
"""


class BiLSTMMaxPoolEncoder(nn.Module):
    def __init__(self, config):
        super(BiLSTMMaxPoolEncoder, self).__init__()
        self.word_emb_dim = config['word_emb_dim']
        self.lstm_dim = config['lstm_dim']
        self.lstm_layers = config['lstm_layers']
        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.lstm_dim, self.lstm_layers,
                                bidirectional=True)

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Network
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*lstm_dim
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)

        # Max pooling
        emb = torch.max(sent_output, 0)[0]

        return emb
