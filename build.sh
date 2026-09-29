#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate llm_course

# python -u scripts/train.py
# python -u api.py
fastapi dev ./api.py
