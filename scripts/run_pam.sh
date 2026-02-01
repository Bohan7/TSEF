#!/bin/bash
# TSEF Attack on PAM dataset - All Explainers

DATASET="pam"
DATA_PATH="../../datasets"

# Model paths (update these paths according to your setup)
TIMEX_MODEL_PATH="/path/to/timex/model_split=1.pt"
TIMEXPP_MODEL_PATH="/path/to/timexpp/model_split=1.pt"
PREDICTOR_PATH="/path/to/classifier/model_split=1.pt"

echo "=========================================="
echo "[1/3] Running TSEF Attack with TimeX"
echo "=========================================="
python ../experiments/evaluation/tsef_attack.py \
    --dataset ${DATASET} \
    --exp_method ours \
    --attack_name tsef_mask_attack \
    --attack 0.1 \
    --data_path ${DATA_PATH} \
    --model_path ${TIMEX_MODEL_PATH} \
    --predictor_path ${PREDICTOR_PATH} \
    --split_no -1 \
    --is_timex \
    --tsef_attack_iters 100 \
    --tsef_attack_k1 10 \
    --tsef_attack_k2 20 \
    --tsef_attack_lr_t 1.0 \
    --tsef_attack_lr_f 1.0 \
    --tsef_attack_r_mt 0.3 \
    --tsef_attack_beta_t 1.0 \
    --tsef_attack_gamma_t 1.0 \
    --tsef_attack_lambda_cls 0.05 \
    --tsef_attack_lambda_exp 1.0 \
    --real_exp_ratio 0.1

echo ""
echo "=========================================="
echo "[2/3] Running TSEF Attack with TimeX++"
echo "=========================================="
python ../experiments/evaluation/tsef_attack.py \
    --dataset ${DATASET} \
    --exp_method ours \
    --attack_name tsef_mask_attack \
    --attack 0.1 \
    --data_path ${DATA_PATH} \
    --model_path ${TIMEXPP_MODEL_PATH} \
    --predictor_path ${PREDICTOR_PATH} \
    --split_no -1 \
    --tsef_attack_iters 100 \
    --tsef_attack_k1 10 \
    --tsef_attack_k2 20 \
    --tsef_attack_lr_t 1.0 \
    --tsef_attack_lr_f 1.0 \
    --tsef_attack_r_mt 0.3 \
    --tsef_attack_beta_t 1.0 \
    --tsef_attack_gamma_t 1.0 \
    --tsef_attack_lambda_cls 0.05 \
    --tsef_attack_lambda_exp 1.0 \
    --real_exp_ratio 0.1

echo ""
echo "=========================================="
echo "[3/3] Running TSEF Attack with IG"
echo "=========================================="
python ../experiments/evaluation/tsef_attack.py \
    --dataset ${DATASET} \
    --exp_method ig_batch \
    --attack_name tsef_mask_attack \
    --attack 0.1 \
    --data_path ${DATA_PATH} \
    --predictor_path ${PREDICTOR_PATH} \
    --split_no -1 \
    --tsef_attack_iters 100 \
    --tsef_attack_k1 10 \
    --tsef_attack_k2 20 \
    --tsef_attack_lr_t 1.0 \
    --tsef_attack_lr_f 1.0 \
    --tsef_attack_r_mt 0.3 \
    --tsef_attack_beta_t 1.0 \
    --tsef_attack_gamma_t 1.0 \
    --tsef_attack_lambda_cls 0.005 \
    --tsef_attack_lambda_exp 1.0 \
    --real_exp_ratio 0.1

echo ""
echo "=========================================="
echo "All PAM experiments completed!"
echo "=========================================="
