#!/usr/bin/env bash
python -m ensurepip --default-pip
pip install numpy
pip install scipy
python3 mnist.py
