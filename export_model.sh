DATASET=cnn_dailymail
# DATASET=narrativeqa
# RUN_NAME=qa_lora_3epoch
# RUN_NAME=cnndm_lora_3epoch
RUN_NAME=alpaca_lora_3epoch
# RUN_NAME=multi_answer_3epoch
OUTPUT_DIR=outputs/Baichuan-7B/$DATASET/$RUN_NAME
CHECKPOINT_NUM=11000
CHECKPOINT_DIR=$OUTPUT_DIR/checkpoint-$CHECKPOINT_NUM
EXPORT_DIR=outputs/export/$DATASET/$RUN_NAME/checkpoint-$CHECKPOINT_NUM

mkdir -p $EXPORT_DIR

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 WANDB_DISABLED=1 \
OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
    --model_name_or_path /home/jqcao/huggingface/Baichuan-7B \
    --template alpaca \
    --finetuning_type lora \
    --checkpoint_dir $CHECKPOINT_DIR \
    --export_dir $EXPORT_DIR