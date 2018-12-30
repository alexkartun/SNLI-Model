import util_nli as ut
import numpy as np
import argparse
import nltk

parser = argparse.ArgumentParser(description='Word vectors generator')

# paths
parser.add_argument("--nli_path", type=str, default='.data/SNLI/snli_1.0/', help="NLI data path")
parser.add_argument("--word_emb_path", type=str, default=".data/GloVe/glove.840B.300d.txt",
                    help="Word embedding file path")
parser.add_argument("--vocab_path", type=str, default='.data/GloVe/vocab.pickle',
                    help="Vocab output path")


def build_vocab(sentences, glove_path):
    print('building vocab from {}...'.format(glove_path))
    word_dict = get_word_dict(sentences)
    vocab = get_glove(word_dict, glove_path)
    print('Done!')
    return vocab


def get_word_dict(sentences):
    word_dict = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    vocab = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                vocab[word] = np.fromstring(vec, sep=' ')
    return vocab


def main():
    config = parser.parse_args()
    train, dev, test = ut.splits(config.nli_path)
    vocab = build_vocab(train['premise'] + train['hypothesis'] + dev['premise'] + dev['hypothesis'] +
                        test['premise'] + test['hypothesis'], config.word_emb_path)

    ut.save_vocab(config.vocab_path, vocab)


if __name__ == '__main__':
    main()
