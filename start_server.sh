#!/bin/bash
source deactivate
conda activate rayhane-ysh
CUDA_VISIBLE_DEVICES=4 gunicorn --bind=0.0.0.0:8003 --workers=1 json_server:app