#!/bin/bash
python eval_llm_csr.py --model_name mistralai/Mistral-7B-v0.1 \  
    --test_file siqa_eng-socmed_eng.json \
    --context translated_context \
    --question translated_question \
    --options "['translated_answerA','translated_answerB', 'translated_answerC']" \
    --answer label