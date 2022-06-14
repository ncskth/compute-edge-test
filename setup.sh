#!/bin/env bash

if [ $# -eq 0 ]; then
    echo "Name required"
    exit 1
fi

apt update
apt install git -y
apt install python3-venv -y

python3 -m venv .venv
. .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python benchmark.py $1