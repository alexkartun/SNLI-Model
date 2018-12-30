import sys
import os
import zipfile

FILE_EXTENSION = 'jsonl'


def download(url, dir_path):

    print('downloading {}...'.format(url))
    file_name = url.split('/')[-1]
    file_path = os.path.join(dir_path, file_name)
    os.system('curl -Lo {} {}'.format(file_path, url))
    print('Done!')
    return file_path


def unzip(file_path, is_snli):

    print("extracting: {}...".format(file_path))
    dir_path = os.path.dirname(file_path)
    with zipfile.ZipFile(file_path, 'r') as z:
        if is_snli:
            for name in z.namelist():
                if name.endswith(FILE_EXTENSION):
                    z.extract(name, dir_path)
        else:
            z.extractall(dir_path)
    os.remove(file_path)
    print('Done!')


def download_snli(dir_path):

    url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    unzip(download(url, dir_path), True)


def download_wordvecs(dir_path):

    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dir_path), False)


def main():

    # define paths
    project_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(project_dir, '.data')
    snli_dir = os.path.join(data_dir, 'SNLI')
    word_vec_dir = os.path.join(data_dir, 'GloVe')

    # create folder structure
    print('creating the folder structure...')
    if os.path.exists(data_dir):
        print('folder structure already existing - skipping...')
        return
    os.makedirs(data_dir)
    os.makedirs(snli_dir)
    os.makedirs(word_vec_dir)
    print('Done!')

    # download SNLI and GloVe data
    download_snli(snli_dir)
    download_wordvecs(word_vec_dir)


if __name__ == '__main__':
    main()
