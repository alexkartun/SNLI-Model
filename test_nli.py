import util_nli as ut
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model_nli import NLINet
import os


"""
ARGUMENTS
"""

parser = argparse.ArgumentParser(description='NLI testing')

# paths
parser.add_argument("--nli_path", type=str, default='.data/SNLI/snli_1.0/', help="NLI data path")
parser.add_argument("--vocab_path", type=str, default=".data/GloVe/vocab.pickle",
                    help="Vocab input path")
parser.add_argument("--output_dir", type=str, default='.output', help="Output directory")
parser.add_argument("--model_name", type=str, default='model.pickle')

# testing
parser.add_argument("--batch_size", type=int, default=64)


# model
parser.add_argument("--lstm_dim", type=int, default=4096, help="lstm hidden state dimension")
parser.add_argument("--lstm_layers", type=int, default=1, help="lstm num layers")
parser.add_argument("--dropout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--mlp_dim", type=int, default=512, help="hidden dim of mlp layers")
parser.add_argument("--output_dim", type=int, default=3, help="entailment/neutral/contradiction")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")


"""
CONFIG
"""


config = parser.parse_args()

"""
DATA
"""


_, _, test = ut.splits(config.nli_path)
word_vectors = ut.load_vocab(config.vocab_path)

for split in ['premise', 'hypothesis']:
    for data_type in ['test']:
        eval(data_type)[split] = np.array([['<s>'] +
                                           [word for word in sent.split() if word in word_vectors] +
                                           ['</s>'] for sent in eval(data_type)[split]])

"""
MODEL
"""


# model's config
config_nli_model = {
    'word_emb_dim': config.word_emb_dim,
    'lstm_dim': config.lstm_dim,
    'lstm_layers': config.lstm_layers,
    'dropout_fc': config.dropout_fc,
    'mlp_dim': config.mlp_dim,
    'output_dim': config.output_dim
}

# model
nli_net = NLINet(config_nli_model)

"""
TEST
"""


def test_model():
    print('TESTING')
    nli_net.eval()

    correct = 0.

    premises = test['premise']
    hypothesises = test['hypothesis']
    targets = test['label']

    for idx in range(0, len(premises), config.batch_size):
        # prepare batch
        premises_batch, premises_len = ut.get_batch(premises[idx:idx + config.batch_size],
                                                    word_vectors, config.word_emb_dim)
        hypothesises_batch, hypothesises_len = ut.get_batch(hypothesises[idx:idx + config.batch_size],
                                                            word_vectors, config.word_emb_dim)
        premises_batch, hypothesises_batch = Variable(premises_batch), Variable(hypothesises_batch)
        targets_batch = Variable(torch.LongTensor(targets[idx:idx + config.batch_size]))

        # model forward
        output = nli_net((premises_batch, premises_len), (hypothesises_batch, hypothesises_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(targets_batch.data.long()).cpu().sum().item()

    # calculate overall accuracy on test
    test_acc = round(100 * correct / len(premises), 2)
    print('accuracy on test {0}'
          .format(test_acc))


"""
Test model on Natural Language Inference task
"""


# Run best model on test set.
nli_net.load_state_dict(torch.load(os.path.join(config.output_dir, config.model_name)))

test_model()
