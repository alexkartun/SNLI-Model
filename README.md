* You should run all the scripts in linux shell on windows platform.
## Dependencies
This code is written in python 3.7. Dependencies include:
* Numpy
* Pytorch For 3.7
* NLTK
This will install all dependencies in pip environment:
```bash
./install.sh
```
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