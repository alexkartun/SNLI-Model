import os
import json
import numpy as np
import pickle
import torch


def splits(data_path):
    """ splitting all data from train/dev/test data sets """
    print('splitting data {}...'.format(data_path))
    train = {}
    dev = {}
    test = {}

    gold_labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    for data_type in ['train', 'dev', 'test']:
        path = os.path.join(data_path, 'snli_1.0_{}.jsonl'.format(data_type))
        with open(path) as f:
            json_data = [json.loads(line) for line in f]

        eval(data_type)['premise'] = [d['sentence1'] for d in json_data if d['gold_label'] != '-']
        eval(data_type)['hypothesis'] = [d['sentence2'] for d in json_data if d['gold_label'] != '-']
        eval(data_type)['label'] = np.array([gold_labels[d['gold_label']] for d in json_data if d['gold_label'] != '-'])

    print('Done!')
    return train, dev, test


def save_vocab(path, vocab):
    """ saving vocabulary of words to file """
    print('saving vocab to {}...'.format(path))
    with open(path, 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')


def load_vocab(path):
    """ loading vocabulary of words from file """
    print('loading vocab from {}...'.format(path))
    with open(path, 'rb') as handle:
        vocab = pickle.load(handle)
    print('Done!')
    return vocab


def get_batch(batch, word_vectors, emb_dim=300):
    """ getting batch of sentences with 3d dim """
    # sent in batch in order of (bsize, max_len, word_dim=300)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vectors[batch[i][j]]
    return torch.from_numpy(embed).float(), lengths
