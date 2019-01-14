import util_nli as ut
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from model_nli import NLINet
import os


"""
ARGUMENTS
"""

parser = argparse.ArgumentParser(description='NLI training')

# paths
parser.add_argument("--nli_path", type=str, default='.data/SNLI/snli_1.0/', help="NLI data path")
parser.add_argument("--vocab_path", type=str, default=".data/GloVe/vocab.pickle",
                    help="Vocab input path")
parser.add_argument("--output_dir", type=str, default='.output', help="Output directory")
parser.add_argument("-model_name", type=str, default='model.pickle')

# training
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dev_every", type=int, default=2)
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")


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


train, dev, _ = ut.splits(config.nli_path)
word_vectors = ut.load_vocab(config.vocab_path)

for split in ['premise', 'hypothesis']:
    for data_type in ['train', 'dev']:
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

# loss
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(nli_net.parameters(), config.learning_rate)

"""
TRAIN
"""


dev_acc_best = -1e10
stop_training = False


def train_model(ep):
    print('TRAINING : Epoch {0}'.format(ep))
    nli_net.train()

    correct = 0.

    premises = train['premise']
    hypothesises = train['hypothesis']
    targets = train['label']

    if ep > 1:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * config.decay

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

        # loss
        loss = loss_fn(output, targets_batch)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # optimizer step
        optimizer.step()

    # calculate overall accuracy on train
    train_acc = round(100 * correct / len(premises), 2)
    print('results : epoch {0} ; accuracy on train {1}'
          .format(ep, train_acc))


def evaluate_model(ep):
    print('VALIDATION : Epoch {0}'.format(ep))
    nli_net.eval()

    correct = 0.
    global dev_acc_best, stop_training

    premises = dev['premise']
    hypothesises = dev['hypothesis']
    targets = dev['label']

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

    # calculate overall accuracy on dev
    eval_acc = round(100 * correct / len(premises), 2)
    print('results : epoch {0} ; accuracy on dev {1}'
          .format(ep, eval_acc))
    
    # check for early stopping
    if eval_acc > dev_acc_best:
        # saving best model for now
        print('saving model at epoch {0}'.format(epoch))
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        torch.save(nli_net.state_dict(), os.path.join(config.output_dir,
                                                      config.model_name))
        dev_acc_best = eval_acc
    else:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / config.lrshrink
        if optimizer.param_groups[0]['lr'] < config.minlr:
            stop_training = True


"""
Train model on NLI task
"""

start = time.time()
epoch = 1

while not stop_training and epoch <= config.epochs:
    train_model(epoch)
    evaluate_model(epoch)
    epoch += 1

end = time.time()
print('training time: {}'.format(str(end - start)))
