#!/bin/bash

# predict
###############
VLM_MODEL='<VLLM_TEST_MODEL_NAME>'
max_retries=3
num_threads=16
timeout=240
max_token_len=8192
test_file_name=mmesci_1019_ja 
# mmesci_1019_img mmesci_1019_zh mmesci_1019_en mmesci_1019_fr mmesci_1019_es mmesci_1019_ja 
eval_mode='image_text_ja' 
# image image_text_zh image_text_en image_text_fr image_text_es image_text_ja

python vllm_localapi_eval.py \
    --test_file_name ${test_file_name} \
    --VLM_MODEL ${VLM_MODEL} \
    --max_retries ${max_retries} \
    --num_threads ${num_threads} \
    --timeout ${timeout} \
    --max_token_len ${max_token_len} \
    --eval_mode ${eval_mode} \
    --w_eval_prompt \
###############


# judge
###############
JUDGE_MODEL='<YOUR_JUDGE_MODEL_NAME>'
test_model_name=${VLM_MODEL}
test_data_name=${test_file_name}
max_retries=5
num_threads=16
timeout=180
max_token_len=256

python get_judge_res.py \
    --JUDGE_MODEL ${JUDGE_MODEL} \
    --test_model_name ${test_model_name} \
    --test_data_name ${test_data_name} \
    --max_retries ${max_retries} \
    --num_threads ${num_threads} \
    --timeout ${timeout} \
    --max_token_len ${max_token_len} \
    --eval_mode ${eval_mode} \
    --w_extraction \
###############


# metrices
###############
python get_metrices_res.py \
    --JUDGE_MODEL ${JUDGE_MODEL} \
    --test_model_name ${test_model_name} \
    --test_data_name ${test_data_name} \
    --eval_mode ${eval_mode} \
###############
