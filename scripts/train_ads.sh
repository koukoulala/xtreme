#!/bin/bash 
# Copyright 2020 Google and DeepMind. 
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
# 
#     http://www.apache.org/licenses/LICENSE-2.0 
# 
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
 
REPO=$PWD 
MODEL=${1:-bert-base-multilingual-cased} 
GPU=${2:-0} 
train_data_path=${3}
eval_data_path=${4}
OUT_DIR=${5:-"$REPO/outputs/"}
model_name_or_path=${6}
 
export CUDA_VISIBLE_DEVICES=$GPU 
 
TASK='ads'
LR=2e-5 
EPOCH=50
MAXL=128 
LANGS="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh" 
LC="" 
if [ $MODEL == "bert-base-multilingual-cased" ]; then 
  MODEL_TYPE="bert" 
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then 
  MODEL_TYPE="xlm" 
  LC=" --do_lower_case" 
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then 
  MODEL_TYPE="xlmr" 
fi 
 
if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then 
  BATCH_SIZE=2 
  GRAD_ACC=16 
  #LR=3e-5
  LR=1e-6
else 
  BATCH_SIZE=8 
  GRAD_ACC=4 
  LR=2e-5 
fi 
 
SAVE_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/" 
mkdir -p $SAVE_DIR 
 
python $PWD/third_party/run_ads.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $model_name_or_path \
  --train_language en \
  --task_name $TASK \
  --do_train \
  --do_eval \
  --do_predict \
  --train_data_path $train_data_path \
  --eval_data_path $eval_data_path \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR/ \
  --save_steps 100 \
  --eval_all_checkpoints \
  --log_file 'train' \
  --predict_languages $LANGS \
  --save_only_best_checkpoint \
  --overwrite_output_dir \
  --eval_test_set $LC \
  --multi_lang
