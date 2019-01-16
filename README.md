### You must run run the scripts on windows platform and linux bash f.e git bash.
## Dependencies
This code is written in python 3.7. Dependencies include:
* Numpy
* Pytorch For 3.7 - Windows
* NLTK
This will install all dependencies in pip environment:
```bash
./install.sh
```
* The 'install.sh' script is not must but you must have all dependencies installed on windows platform.
### From here you need to run all the scripts.
## Fetch and preprocess the data
This will download SNLI datasets and GloVe word pretrained vectors and
preprocess vocabulary:
```bash
./download_and_preprocess.sh
```
## Train model on Natural Language Inference (NLI)
```bash
./train.sh
```
## Test model on Natural Language Inference (NLI)
```bash
./test.sh
```
## Contributors
```
title     = {Supervised Learning of NLI task}
authors    = {Alex Kartun, Ofir Sharon}
```
