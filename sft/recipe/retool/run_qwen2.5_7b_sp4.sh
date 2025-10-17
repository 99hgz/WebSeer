#!/bin/bash
set -x

export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
WANDB_MODE=offline
ulimit -n 65535

EXPERIMENT_NAME=re_rag_sft-qwen2.5-3b-1.5-sp4-mb

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.max_length=16384 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=8 \
    data.train_files=$HOME/data/re_rag/train.parquet \
    data.val_files=$HOME/data/re_rag/test.parquet \
    +data.filter_overlong_prompts=True \
    data.truncation=left \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    model.partial_pretrain=/data2/MODELS0/Qwen2.5-3B-Instruct \
    model.trust_remote_code=true \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.offload_params=false \
    optim.lr=1e-6 \
    trainer.default_local_dir=/data3/hgz/checkpoints/re_rag_sft/$EXPERIMENT_NAME \
    trainer.project_name=re_rag_sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=4 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true
