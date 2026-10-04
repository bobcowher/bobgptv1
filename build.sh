#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate llm_course

# python -u scripts/pretrain.py
# python -u api.py
fastapi dev ./api.py --host 0.0.0.0
