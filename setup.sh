#!/bin/bash

printf "Initializing python virtualenv...\n\n"
virtualenv -p python3.5 env
. env/bin/activate

printf "Installing dependencies...\n\n"
pip install -r requirements.txt
pip install torch

printf "Installing submodules...\n\n"
git submodule init
git submodule update

printf "Installing nltk...\n\n"
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet');"

printf "Initializing python virtualenv...\n\n"
deactivate
virtualenv -p python2.7 utils/env
. utils/env/bin/activate

printf "Installing dependencies...\n\n"
pip install -r utils/requirements.txt