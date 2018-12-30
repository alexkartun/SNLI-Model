#!/bin/bash
### downloading and preprocessing data
echo -e "downloading data..."
python download.py
echo -e "Done!"
echo -e "preprocessing data..."
python preprocess.py
echo -e "Done!"