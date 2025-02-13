# 8.26
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_predict --do_eval --pad_to_max_length &> logs/xnli_en.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_predict --do_eval --pad_to_max_length &> logs/xnli_en.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="xlm-roberta-large-finetuned-conll03-english"  --language=en  --output_dir=./results/xnli_en --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-en --do_predict --do_eval --pad_to_max_length &> logs/xnli_en.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u examples/pytorch/text-classification/run_xnli_each.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --predict_file=../datasets/DescAndSentence/try_1 --cache_dir=../ckpt/xlm-roberta-large-xnli &> logs/xnli_try.out &
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_train --do_predict --do_eval --pad_to_max_length --num_train_epochs=3 &> logs/xnli_en.out &
nohup python -u examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_train --do_predict --do_eval --pad_to_max_length --num_train_epochs=3 &> logs/xnli_en.out &

"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_xnli && . /tmp/env_xnli/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/summarization/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u examples/pytorch/text-classification/run_xnli_each.py  --model_name_or_path=joeddav/xlm-roberta-large-xnli  --language=en  --output_dir=[#output-model-path] --predict_file=[#input-training-data-path] --cache_dir=[#input-previous-model-path] &"
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_xnli_en && . /tmp/env_xnli_en/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/text-classification/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u -m torch.distributed.launch --nproc_per_node 4 --use_env examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path=joeddav/xlm-roberta-large-xnli  --language=en  --output_dir=./results/xnli_train_en --per_device_train_batch_size=4 --per_device_eval_batch_size=8 --cache_dir=[#input-previous-model-path] --do_train --do_predict --do_eval --pad_to_max_length --num_train_epochs=30 &"

# 8.27
"pip install --user . && pip install --user numpy==1.20.0 && pip install -r ./examples/pytorch/summarization/requirements.txt && python -u examples/pytorch/text-classification/run_xnli_each.py  --model_name_or_path=joeddav/xlm-roberta-large-xnli  --language=en  --output_dir=[#output-model-path] --predict_file=[#input-training-data-path] --cache_dir=[#input-previous-model-path] &"

nohup bash ./scripts/train_xnli.sh xlm-roberta-large 1 &> logs/try.out &
CUDA_VISIBLE_DEVICES=1 nohup bash ./scripts/train_xnli_2.sh joeddav/xlm-roberta-large-xnli 1 &> logs/try_2.out &

"cd ../xtreme_transformer/transformers && pip install --user . && cd ../../xtreme && bash ./scripts/train_xnli_2.sh xlm-roberta-large [#input-training-data-path] [#output-model-path] [#input-previous-model-path]"

# 8.30
"cd ../xtreme_transformer/transformers && pip install --user . && cd ../../xtreme && bash ./scripts/train.sh xlm-roberta-large xnli 0 [#input-training-data-path] [#output-model-path]"

# 9.7
"cd ../xtreme_transformer/transformers && pip install --user . && cd ../../xtreme && bash ./scripts/train_ads.sh xlm-roberta-large 0 [#input-training-data-path]/adsnli_train_en.tsv [#input-training-data-path] [#output-model-path] [#input-previous-model-path]"

# 9.7
"cd ../xtreme_transformer/transformers && pip install --user . && cd ../../xtreme && bash ./scripts/train_ads.sh xlm-roberta-large 0 [#input-training-data-path]/adsnli_train_en_head_des.tsv [#input-training-data-path] [#output-model-path] [#input-previous-model-path]"

# 9.13
"cd ../xtreme_transformer/transformers && pip install --user . && cd ../../xtreme && bash ./scripts/train_ads.sh xlm-roberta-large 0 [#input-training-data-path]/multi_train.tsv [#input-training-data-path] [#output-model-path] [#input-previous-model-path]"

# 9.26
"cd ../xtreme_transformer/transformers && pip install --user . && cd ../../xtreme && bash ./scripts/evaluate_ads.sh xlm-roberta-large 0 [#input-training-data-path] [#input-training-data-path] [#output-model-path] [#input-previous-model-path]"
