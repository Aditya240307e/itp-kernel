#!/usr/bin/env bash
# 03_pretrain_finetune.sh -- behaviour cloning, per seed:
#     pre/n<N>   pre-training on N synthetic theorems only (N in synth_sizes: the scaling study)
#     ft/bc      pre-trained (N = max) then fine-tuned on the bank's train split          ft/bc_scratch   the same without pre-training
#     ft/bc_mw   manager-worker labels           ft/bc_n<N>   fine-tuned after a smaller pre-training
# DEVICE=cuda trains on the GPU (by far the biggest saving: the CPU does ~55 steps/s on the reference machine).
. "$(dirname "$0")/env.sh"
stage_script 03_pretrain_finetune pre,ft "3. pre-training and fine-tuning"
