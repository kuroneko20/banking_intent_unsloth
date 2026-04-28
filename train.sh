#!/bin/bash
pip install setfit -q
python scripts/preprocess_data.py
python scripts/train.py