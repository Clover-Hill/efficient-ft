RUN_NAME=cnndm_full_1epoch
DATASET=cnn_dailymail
MODEL=/home/jqcao/huggingface/Baichuan-7B
PROJECT_NAME=Baichuan-7B-finetune
OUTPUT_DIR=outputs/Baichuan-7B/$DATASET/$RUN_NAME

mkdir -p $OUTPUT_DIR

cp full_cnndm_baichuan.sh $OUTPUT_DIR/run.sh

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1 WANDB_PROJECT=$PROJECT_NAME deepspeed --include="localhost:0,1,2,3"  \
    --master_port='30306' src/train_bash.py \
    --stage sft \
    --model_name_or_path $MODEL \
    --do_train \
    --dataset cnn_dailymail \
    --preprocessing_num_workers 100 \
    --template alpaca \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --cutoff_len 1024 \
    --max_new_tokens 64 \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --run_name $RUN_NAME \
    --overwrite_output_dir True \
    --deepspeed ds_config/ds_config.json \
| tee -a $OUTPUT_DIR/train.log
