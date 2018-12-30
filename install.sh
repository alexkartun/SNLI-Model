#!/bin/bash
### install dependencies
echo -e "installing dependencies..."
pip3 install numpy
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-win_amd64.whl
pip3 install torchvision
pip3 install nltk
python -c "import nltk; nltk.download('punkt')"
echo -e "Done!"