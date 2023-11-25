DATASET=cnn_dailymail
RUN_NAME=alpaca_lora_3epoch
OUTPUT_DIR=outputs/export/$DATASET/$RUN_NAME
CHECKPOINT_NUM=11000
CHECKPOINT_DIR=$OUTPUT_DIR/checkpoint-$CHECKPOINT_NUM
MODEL=/home/jqcao/huggingface/Baichuan-7B

mkdir -p $OUTPUT_DIR

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 WANDB_DISABLED=1 \
OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ./ds_config/default_config.json \
    src/train_bash.py \
    --stage sft \
    --model_name_or_path $CHECKPOINT_DIR \
    --do_predict \
    --preprocessing_num_workers 100 \
    --dataset cnn_dailymail_test \
    --template alpaca \
    --finetuning_type lora \
    --output_dir $OUTPUT_DIR \
    --bf16 \
    --cutoff_len 1024 \
    --max_length 64 \
    --num_beams 4 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
| tee -a $OUTPUT_DIR/eval.log