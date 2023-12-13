#!/bin/bash
python eval_fluency_english_socmed_csr_data.py \
    --do_train \ 
    --do_predict \ 
    --checkpoints_dir ./checkpoints-roberta-classifier \ 
    --pair_fields "[('context','translated_context'), ('question','translated_question'), ('answerA', 'translated_answerA'), ('answerB', 'translated_answerB'), ('answerC', 'translated_answerC')]" \
    --classifier_train_data ./datasets/classifier/train-socmed-classification.json \
    --classifier_val_data ./datasets/classifier/val-socmed-classification.json \ 
    --test_file ./datasets/csr/siqa_eng-socmed_eng.json