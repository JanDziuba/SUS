#!/usr/bin/env bash

python3 -m venv myVenv

source myVenv/bin/activate

pip3 install -r requirements.txt

python3 main.py "$1"