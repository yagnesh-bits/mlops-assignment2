#!/bin/bash
# Test the training command that DVC will run

set -x

python -m src.models.train \
  --data_dir data/processed \
  --output_dir models/artifacts \
  --model_type cnn \
  --epochs 2 \
  --batch_size 32 \
  --lr 0.001 \
  --mlflow_uri http://localhost:5000

