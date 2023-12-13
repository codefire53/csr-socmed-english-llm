#!/bin/bash
open_ai_api_key="<please-insert-your-key-here>"
export OPENAI_API_KEY=$open_ai_api_key
python create_socmed_english_csr.py \
    --dataset_name siqa \
    --batch_size 30