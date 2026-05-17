"""
TSEF: Time Series Explanation Fooler

This module implements the TSEF attack methods for evaluating the robustness of 
time-series explanation models. The main attack functions are:
- tsef_mask_attack: TSEF attack for mask-based explainers (e.g., TimeX)
- tsef_mask_attack_ig: TSEF attack for gradient-based explainers (e.g., Integrated Gradients)

"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*xavier_uniform.*")

import os
import re
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.encoders.simple import CNN, LSTM
from txai.utils.experimental import get_explainer
from txai.vis.vis_saliency import vis_one_saliency
from txai.utils.data import process_Synth
from txai.utils.data.preprocess import process_MITECG
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.preprocess import process_Epilepsy, process_PAM

from txai.utils.evaluation import (
    target_ground_truth_xai_eval, target_ground_truth_IoU, 
    ground_truth_xai_eval, ground_truth_IoU, 
    topk_intersection, dtw_distance, l2_distance, 
    topk_intersection_10, topk_intersection_5, 
    topk_intersection_30, topk_intersection_50, 
    rank_order_correlation, mae_exp
)
from txai.utils.functional import js_divergence
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.loss import GSATLoss, ConnectLoss, ConnectLoss_Mt
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.predictors import eval_mvts_transformer_asr

# Import both model types - selection will be done via command-line args
from txai.models.bc_model4 import TimeXModel as TimeXModel_v4
from txai.models.bc_model import TimeXModel as TimeXModel_base

import pickle

# Default value (can be overridden by command-line args)
is_timex = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_random_target_label(device, labels=None, n_classes=4):
    """
    Generate random target labels different from the original labels.
    
    Args:
        device: torch device
        labels: Original labels tensor
        n_classes: Number of classes
        
    Returns:
        target_labels: Random target labels different from original
    """
    target_labels = torch.zeros_like(labels)
    for counter in range(labels.shape[0]):
        l = list(range(n_classes))
        l.remove(labels[counter])
        t = (len(l) * torch.rand([1])).long().to(device)
        target_labels[counter] = l[t]
    return target_labels.long().to(device)


# =============================================================================
# TSEF Attack Helper Functions
# =============================================================================

def gumbel_sigmoid(total_mask, tau=1.0, hard=True):
    """
    Gumbel-Sigmoid reparameterization for differentiable binary masks.
    
    Args:
        total_mask: Probability mask tensor
        tau: Temperature parameter for Gumbel-Softmax
        hard: If True, use straight-through estimator for hard samples
        
    Returns:
        Reparameterized mask
    """
    inv_probs = 1 - total_mask
    total_mask_prob = torch.stack([inv_probs, total_mask], dim=-1)
    total_mask_reparameterize = F.gumbel_softmax(
        torch.log(total_mask_prob + 1e-9), tau=tau, hard=hard
    )[..., 1]
    return total_mask_reparameterize


def compute_alpha_for_theta_f(
    ori_X,         # [T, B, D]  original clean signal
    Mt_used,       # [T, B, D]  temporal mask in [0,1]
    theta_f,       # [Fbins, B, D] frequency params (real), Mf = 1 + α * tanh(theta_f)
    eps_prime,     # scalar, [1,B,1], [1,B,D], [B,1], [B,D], or [1] -- we handle/broadcast
    *,
    gamma=0.98,    # safety factor (<1 to avoid tiny overshoots)
    max_alpha=5.0, # clamp to prevent runaway scaling on near-zero windows (None to disable)
    per_channel=True,  # True: α per (sample,channel); False: one α per sample (satisfies all channels)
    tiny=1e-12,
    norm='ortho'   # must match your rfft/irfft calls
):
    """
    Compute adaptive alpha scaling for frequency mask to ensure epsilon budget.
    
    This function computes alpha such that the L-infinity norm of the perturbation
    induced by the frequency mask stays within the epsilon budget.
    
    Args:
        ori_X: Original input signal [T, B, D]
        Mt_used: Temporal mask [T, B, D]
        theta_f: Frequency mask parameters [Fbins, B, D]
        eps_prime: Epsilon budget (scalar or per-sample/channel)
        gamma: Safety factor to avoid overshoots
        max_alpha: Maximum alpha value to prevent instability
        per_channel: If True, compute alpha per (sample, channel)
        tiny: Small constant for numerical stability
        norm: FFT normalization mode
        
    Returns:
        alpha_freq: Alpha broadcasted to frequency shape [Fbins, B, D]
        alpha_sample: Alpha per sample [1, B, D]
        delta_base: Base perturbation for alpha=1 [T, B, D]
    """
    if ori_X.shape != Mt_used.shape:
        raise ValueError("ori_X and Mt_used must have identical shape [T,B,D].")
    if ori_X.ndim != 3:
        raise ValueError(f"Expected ori_X to be [T,B,D], got {ori_X.shape}.")

    T, B, D = ori_X.shape
    Fbins = theta_f.shape[0]

    # Window and FFT
    W = Mt_used * ori_X                                      # [T,B,D]
    W_fft = torch.fft.rfft(W, dim=0, norm=norm)              # [Fbins,B,D]
    if theta_f.shape != W_fft.shape:
        raise ValueError(f"theta_f shape {theta_f.shape} must match rfft(Mt*ori_X) shape {W_fft.shape}.")

    # Δx for α = 1  (real time-domain change from unit-strength frequency offset)
    Tcoef = torch.tanh(theta_f)                              # [Fbins,B,D], real
    delta_base = torch.fft.irfft(Tcoef * W_fft, n=T, dim=0, norm=norm)  # [T,B,D]

    # ||Δx(1)||_∞ over time → per-sample-per-channel base magnitude
    base_max_ch = delta_base.abs().amax(dim=0, keepdim=True)            # [1,B,D]
    base_max_ch = base_max_ch.clamp_min(tiny)

    # Prepare eps' → broadcast/align to [1,B,D]
    eps_t = torch.as_tensor(eps_prime, device=ori_X.device, dtype=ori_X.dtype)
    eps_t = eps_t + torch.zeros((1, B, D), device=ori_X.device, dtype=ori_X.dtype)

    if per_channel:
        alpha_sample = gamma * eps_t / base_max_ch           # [1,B,D]
    else:
        alpha_per_ch = gamma * eps_t / base_max_ch           # [1,B,D]
        alpha_sample = alpha_per_ch.amin(dim=-1, keepdim=True).expand(1, B, D)

    # clamp for stability
    if max_alpha is not None:
        alpha_sample = alpha_sample.clamp(max=max_alpha)
    alpha_sample = alpha_sample.clamp_min(0.0).detach()

    # Expand α to frequency shape
    alpha_freq = alpha_sample.expand(Fbins, -1, -1).contiguous()        # [Fbins,B,D]

    return alpha_freq, alpha_sample, delta_base


# =============================================================================
# TSEF Main Attack Functions
# =============================================================================

def tsef_mask_attack(
    predictor, model, X, gt_x, labels, times, 
    eps=0.3, lambda_cls=0.1, lambda_exp=0.1, iters=1, 
    min_X=-2.9198, max_X=2.9489,
    r_mt=0.5,  # sparsity hyperparameter for temporal mask
    k1=10, k2=10, lr_t=1.0, lr_f=1.0,
    beta_t=2.0, gamma_t=1.0  # sparsity and connectivity loss weights
):
    """
    TSEF Mask Attack for mask-based time-series explainers.
    
    This attack generates adversarial perturbations that fool both the classifier
    and the explainer while preserving the temporal structure of time-series.
    
    The attack operates in two phases:
    1. Temporal Mask (Mt) optimization: Identifies important time regions to perturb
    2. Frequency Mask (Mf) optimization: Applies frequency-domain perturbations
    
    Args:
        predictor: Classification model
        model: Explanation model (e.g., TimeX)
        X: Input time-series [T, B, D]
        gt_x: Ground truth explanations [T, B, D]
        labels: Target labels for attack [B]
        times: Time indices [T, B]
        eps: Perturbation budget (epsilon)
        lambda_cls: Weight for classification loss
        lambda_exp: Weight for explanation loss
        iters: Number of outer iterations
        min_X, max_X: Clipping bounds for adversarial examples
        r_mt: Sparsity ratio for temporal mask
        k1: Inner iterations for Mt optimization
        k2: Inner iterations for Mf optimization
        lr_t: Learning rate for temporal mask
        lr_f: Learning rate for frequency mask
        beta_t: Weight for temporal mask sparsity loss
        gamma_t: Weight for connectivity loss
        
    Returns:
        X: Adversarial examples [T, B, D]
    """
    # For LSTM/CNN models, need train() mode for cudnn backward
    if isinstance(predictor, (CNN, LSTM)):
        predictor.train()
    else:
        predictor.eval()
    model.eval()

    X = X.to(device)
    gt_x = gt_x.to(device)
    T, B, d = X.shape

    # Loss functions
    prd_loss = nn.CrossEntropyLoss()
    reg_loss_mt = GSATLoss(r=r_mt)
    connected_loss = ConnectLoss_Mt()
    explainer_loss = nn.BCELoss()
        
    ori_X = X.detach().clone()

    # FFT of original signal
    ori_X_fft = torch.fft.rfft(ori_X, dim=0, norm='ortho')
    Fbins = ori_X_fft.shape[0]

    for outer in range(iters):
        # Initialize trainable parameters
        theta_t = torch.rand(T, B, d, requires_grad=True, device=device)
        theta_f = torch.rand(Fbins, B, d, requires_grad=True, device=device)
        
        # Phase A: Temporal Mask (Mt) Optimization
        for i in range(k1):
            Mt = gumbel_sigmoid(theta_t.sigmoid(), tau=1.0, hard=True)
            Xadv = (1 - Mt) * X

            out_dict_adv = model.get_saliency_explanation(Xadv, times, captum_input=False)
            outputs_adv = predictor(Xadv, times, captum_input=False)

            prd_cost_mt = prd_loss(outputs_adv, labels.detach())
            explainer_cost_mt = explainer_loss(out_dict_adv['mask_in'], gt_x.to(device))
            sparse_loss_mt = reg_loss_mt(theta_t.sigmoid())
            connected_loss_mt = connected_loss(theta_t.sigmoid())

            J_mt = (lambda_exp * explainer_cost_mt + lambda_cls * prd_cost_mt + 
                   beta_t * sparse_loss_mt + gamma_t * connected_loss_mt)

            model.zero_grad()
            predictor.zero_grad()
            if theta_t.grad is not None:
                theta_t.grad.zero_()

            J_mt.backward()

            # PGD-style sign step
            with torch.no_grad():
                theta_t = theta_t - lr_t * theta_t.grad.sign()
            
            if i % 5 == 0:
                sparse_mt = (Mt.sum() / Mt.flatten().shape[0]).item()
                print(f'[Outer {outer+1}/{iters}] Phase A (Mt) iter {i:3d}/{k1} | '
                      f'Loss: {J_mt.item():7.4f} | '
                      f'Exp: {explainer_cost_mt.item():6.4f} | '
                      f'Cls: {prd_cost_mt.item():6.4f} | '
                      f'Sparse: {sparse_mt:.2%}')
            
            theta_t = theta_t.detach().requires_grad_(True)

        # Freeze temporal mask for frequency block
        Mt_used = Mt.detach()
        X_fft = torch.fft.rfft(Mt_used * X, dim=0, norm='ortho')
        Fbins = X_fft.shape[0]

        # Phase B: Frequency Mask (Mf) Optimization
        for j in range(k2):
            alpha_freq, _, _ = compute_alpha_for_theta_f(
                ori_X=ori_X,
                Mt_used=Mt_used,
                theta_f=theta_f,
                eps_prime=eps,
                gamma=0.98,
                max_alpha=5.0,
                per_channel=True,
                tiny=1e-12,
                norm='ortho'
            )

            Mf = 1 + alpha_freq * theta_f.tanh()
            Mf = Mf.clamp(0.0, 2.0)

            spec = Mf * X_fft
            mf_X = torch.fft.irfft(spec, n=T, dim=0, norm='ortho')

            Xadv = Mt_used * mf_X + (1 - Mt_used) * X
            eta = torch.clamp(Xadv - ori_X, min=-eps, max=eps)
            Xadv = torch.clamp(ori_X + eta, min=min_X, max=max_X)

            out_dict_mf = model.get_saliency_explanation(Xadv, times, captum_input=False)
            outputs_mf = predictor(Xadv, times, captum_input=False)

            prd_cost_mf = prd_loss(outputs_mf, labels)
            explainer_cost_mf = explainer_loss(out_dict_mf['mask_in'], gt_x.to(device))

            J_mf = lambda_exp * explainer_cost_mf + lambda_cls * prd_cost_mf

            model.zero_grad()
            predictor.zero_grad()
            if theta_f.grad is not None:
                theta_f.grad.zero_()

            J_mf.backward()

            # PGD-style sign step
            with torch.no_grad():
                theta_f = theta_f - lr_f * theta_f.grad.sign()

            if j % 5 == 0:
                sparse_mf = (Mf.sum() / Mf.flatten().shape[0]).item()
                print(f'[Outer {outer+1}/{iters}] Phase B (Mf) iter {j:3d}/{k2} | '
                      f'Loss: {J_mf.item():7.4f} | '
                      f'Exp: {explainer_cost_mf.item():6.4f} | '
                      f'Cls: {prd_cost_mf.item():6.4f} | '
                      f'Scale: {sparse_mf:.4f}')
            
            theta_f = theta_f.detach().requires_grad_(True)

        # Combine perturbations and project to epsilon ball
        adv_X = Mt_used * mf_X + (1 - Mt_used) * X
        eta = torch.clamp(adv_X - ori_X, min=-eps, max=eps)
        X = torch.clamp(ori_X + eta, min=min_X, max=max_X).detach_()

    return X


def tsef_mask_attack_ig(
    predictor, X, gt_x, labels, original_predictions, times,
    eps=0.3, lambda_cls=0.1, lambda_exp=0.1, iters=1,
    min_X=-2.9198, max_X=2.9489,
    r_mt=0.5,  # sparsity hyperparameter for temporal mask
    k1=10, k2=10, lr_t=1.0, lr_f=1.0,
    beta_t=2.0, gamma_t=1.0  # sparsity and connectivity loss weights
):
    """
    TSEF Mask Attack for gradient-based explainers (e.g., Integrated Gradients).

    Similar to tsef_mask_attack, but uses IG instead of mask-based explanations.

    Args:
        predictor: Classification model
        X: Input time-series [T, B, D]
        gt_x: Ground truth explanations [T, B, D]
        labels: Target labels for attack [B]
        original_predictions: Predictor's predictions on the clean (unattacked) input, used as the IG target [B]
        times: Time indices [T, B]
        eps: Perturbation budget (epsilon)
        lambda_cls: Weight for classification loss
        lambda_exp: Weight for explanation loss
        iters: Number of outer iterations
        min_X, max_X: Clipping bounds for adversarial examples
        r_mt: Sparsity ratio for temporal mask
        k1: Inner iterations for Mt optimization
        k2: Inner iterations for Mf optimization
        lr_t: Learning rate for temporal mask
        lr_f: Learning rate for frequency mask
        beta_t: Weight for temporal mask sparsity loss
        gamma_t: Weight for connectivity loss
        
    Returns:
        X: Adversarial examples [T, B, D]
    """
    # For LSTM/CNN models, need train() mode for cudnn backward
    if isinstance(predictor, (CNN, LSTM)):
        predictor.train()
    else:
        predictor.eval()

    X = X.to(device)
    gt_x = gt_x.to(device)
    T, B, d = X.shape

    # Loss functions
    prd_loss = nn.CrossEntropyLoss()
    reg_loss_mt = GSATLoss(r=r_mt)
    connected_loss = ConnectLoss_Mt()
    explainer_loss = nn.MSELoss()
        
    ori_X = X.detach().clone()

    # FFT of original signal
    ori_X_fft = torch.fft.rfft(ori_X, dim=0, norm='ortho')
    Fbins = ori_X_fft.shape[0]

    # Create IG object
    from captum.attr import IntegratedGradients
    IG_batch = IntegratedGradients(predictor)
    print("Created batched IG explainer")

    for outer in range(iters):
        # Initialize trainable parameters
        theta_t = torch.rand(T, B, d, requires_grad=True, device=device)
        theta_f = torch.rand(Fbins, B, d, requires_grad=True, device=device)
        
        # Phase A: Temporal Mask (Mt) Optimization
        for i in range(k1):
            Mt = gumbel_sigmoid(theta_t.sigmoid(), tau=1.0, hard=True)
            Xadv = (1 - Mt) * X
            outputs_adv = predictor(Xadv, times, captum_input=False)
            prd_cost_mt = prd_loss(outputs_adv, labels.detach())

            # Compute IG for adversarial batch
            Xadv_batch = Xadv.permute(1, 0, 2)
            times_batch = times.permute(1, 0)
            
            generated_exps_batch = IG_batch.attribute(
                Xadv_batch,
                target=original_predictions,
                additional_forward_args=(times_batch, None, True),
                n_steps=1,
                internal_batch_size=None
            )
            generated_exps = generated_exps_batch.permute(1, 0, 2).float()
            
            explainer_cost_mt = explainer_loss(generated_exps.relu(), gt_x)
            sparse_loss_mt = reg_loss_mt(theta_t.sigmoid())
            connected_loss_mt = connected_loss(theta_t.sigmoid())

            J_mt = (lambda_exp * explainer_cost_mt.float() + lambda_cls * prd_cost_mt + 
                   beta_t * sparse_loss_mt + gamma_t * connected_loss_mt)

            predictor.zero_grad()
            if theta_t.grad is not None:
                theta_t.grad.zero_()

            J_mt.backward()

            # PGD-style sign step
            with torch.no_grad():
                theta_t = theta_t - lr_t * theta_t.grad.sign()
            
            if i % 5 == 0:
                sparse_mt = (Mt.sum() / Mt.flatten().shape[0]).item()
                print(f'[Outer {outer+1}/{iters}] Phase A (Mt-IG) iter {i:3d}/{k1} | '
                      f'Loss: {J_mt.item():7.4f} | '
                      f'Exp: {explainer_cost_mt.item():6.4f} | '
                      f'Cls: {prd_cost_mt.item():6.4f} | '
                      f'Sparse: {sparse_mt:.2%}')
            
            theta_t = theta_t.detach().requires_grad_(True)

        # Freeze temporal mask for frequency block
        Mt_used = Mt.detach()
        X_fft = torch.fft.rfft(Mt_used * X, dim=0, norm='ortho')
        Fbins = X_fft.shape[0]

        # Phase B: Frequency Mask (Mf) Optimization
        for j in range(k2):
            alpha_freq, _, _ = compute_alpha_for_theta_f(
                ori_X=ori_X,
                Mt_used=Mt_used,
                theta_f=theta_f,
                eps_prime=eps,
                gamma=0.98,
                max_alpha=5.0,
                per_channel=True,
                tiny=1e-12,
                norm='ortho'
            )

            Mf = 1 + alpha_freq * theta_f.tanh()
            Mf = Mf.clamp(0.0, 2.0)

            spec = Mf * X_fft
            mf_X = torch.fft.irfft(spec, n=T, dim=0, norm='ortho')

            Xadv = Mt_used * mf_X + (1 - Mt_used) * X
            eta = torch.clamp(Xadv - ori_X, min=-eps, max=eps)
            Xadv = torch.clamp(ori_X + eta, min=min_X, max=max_X)

            # Compute IG for Mf block
            Xadv_batch_mf = Xadv.permute(1, 0, 2)
            times_batch_mf = times.permute(1, 0)
            
            generated_exps_batch_mf = IG_batch.attribute(
                Xadv_batch_mf,
                target=original_predictions,
                additional_forward_args=(times_batch_mf, None, True),
                n_steps=1,
                internal_batch_size=None
            )
            generated_exps_mf = generated_exps_batch_mf.permute(1, 0, 2).float()
            
            outputs_mf = predictor(Xadv, times, captum_input=False)

            prd_cost_mf = prd_loss(outputs_mf, labels)
            explainer_cost_mf = explainer_loss(generated_exps_mf.relu(), gt_x)

            J_mf = lambda_exp * explainer_cost_mf + lambda_cls * prd_cost_mf

            predictor.zero_grad()
            if theta_f.grad is not None:
                theta_f.grad.zero_()

            J_mf.backward()

            # PGD-style sign step
            with torch.no_grad():
                theta_f = theta_f - lr_f * theta_f.grad.sign()

            if j % 5 == 0:
                sparse_mf = (Mf.sum() / Mf.flatten().shape[0]).item()
                print(f'[Outer {outer+1}/{iters}] Phase B (Mf-IG) iter {j:3d}/{k2} | '
                      f'Loss: {J_mf.item():7.4f} | '
                      f'Exp: {explainer_cost_mf.item():6.4f} | '
                      f'Cls: {prd_cost_mf.item():6.4f} | '
                      f'Scale: {sparse_mf:.4f}')
            
            theta_f = theta_f.detach().requires_grad_(True)

        # Combine perturbations and project to epsilon ball
        adv_X = Mt_used * mf_X + (1 - Mt_used) * X
        eta = torch.clamp(adv_X - ori_X, min=-eps, max=eps)
        X = torch.clamp(ori_X + eta, min=min_X, max=max_X).detach_()

    return X


# =============================================================================
# Model and Evaluation Helper Functions
# =============================================================================

def get_model(args, X):
    """
    Create and return the appropriate model based on args.
    
    Args:
        args: Command-line arguments
        X: Input data for determining input dimensions
        
    Returns:
        model: The classification model
    """
    if args.model_type == "cnn":
        model = CNN(
            d_inp=X.shape[-1], 
            n_classes=2 if (args.dataset == "mitecg_hard" or args.dataset == 'anomaly') else 4,
            dim=64 if args.dataset == "anomaly" else 32
        )
    elif args.model_type == "lstm":
        model = LSTM(d_inp=X.shape[-1], n_classes=2 if args.dataset == "mitecg_hard" else 4)
    else:  # transformer
        if args.dataset == 'scs_better':
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                n_classes=4,
                nlayers=2,
                nhead=1,
                trans_dim_feedforward=64,
                trans_dropout=0.25,
                d_pe=16,
            )
        elif args.dataset == 'seqcombsingle':
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                n_classes=4,
                nlayers=2,
                nhead=1,
                trans_dim_feedforward=64,
                trans_dropout=0.25,
                d_pe=16,
            )
        elif args.dataset == 'scs_inline':
            model = TransformerMVTS(
                d_inp=1,
                max_len=200,
                n_classes=4,
                nlayers=2,
                nhead=1,
                trans_dim_feedforward=128,
                trans_dropout=0.2,
                d_pe=16,
            )
        elif args.dataset == 'scs_fixone':
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=2,
                n_classes=4,
                trans_dim_feedforward=32,
                trans_dropout=0.1,
                d_pe=16,
            )
        elif args.dataset == 'seqcomb_mv':
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=2,
                n_classes=4,
                trans_dim_feedforward=128,
                trans_dropout=0.25,
                d_pe=16,
            )
        elif args.dataset == 'mitecg_hard':
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=1,
                n_classes=2,
                trans_dim_feedforward=64,
                trans_dropout=0.1,
                d_pe=16,
                stronger_clf_head=False,
                norm_embedding=True,
            )
        elif args.dataset == 'lowvardetect':
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=1,
                n_classes=4,
                trans_dim_feedforward=32,
                trans_dropout=0.25,
                d_pe=16,
                stronger_clf_head=False,
            )
        elif args.dataset == 'epilepsy':
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                n_classes=2,
                nlayers=1,
                trans_dim_feedforward=16,
                trans_dropout=0.1,
                d_pe=16,
                norm_embedding=False,
            )
        elif args.dataset == 'pam':
            model = TransformerMVTS(
                d_inp=X.shape[2],
                max_len=X.shape[0],
                n_classes=8,
            )
        else:
            # Default transformer configuration
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                n_classes=4,
                trans_dim_feedforward=64,
                trans_dropout=0.25,
                d_pe=16,
            )
    return model


def topk_mask_overall(exp: torch.Tensor, k: int, by_abs: bool = False):
    """
    Create a mask for top-k entries per sample.
    
    Args:
        exp: Saliency tensor [T, B, d]
        k: Number of top entries to select
        by_abs: If True, select by absolute value
        
    Returns:
        mask: Binary mask [T, B, d]
    """
    T, B, d = exp.shape
    scores = exp.permute(1, 0, 2).contiguous().view(B, T * d)
    if by_abs:
        scores = scores.abs()

    k = int(max(0, min(k, T * d)))
    mask_flat = torch.zeros(B, T * d, device=exp.device, dtype=exp.dtype)
    if k > 0:
        topk_idx = torch.topk(scores, k, dim=1, largest=True).indices
        mask_flat.scatter_(1, topk_idx, 1.0)

    mask = mask_flat.view(B, T, d).permute(1, 0, 2).contiguous()
    return mask


def top_k_exp(exp: torch.Tensor, frac: float, **kwargs):
    """Convenience wrapper for topk_mask_overall with fractional k."""
    T, B, d = exp.shape
    k = max(1, int(round(frac * T * d)))
    return topk_mask_overall(exp, k, **kwargs)


def topk_elements_TBD(
    exp_TBD: torch.Tensor,
    rho: float = 0.1,
) -> torch.Tensor:
    """
    Flatten T×D, select top-k% (t,d) positions per sample.
    
    Args:
        exp_TBD: Explanation weights, shape (T, B, D)
        rho: Ratio of elements to select (0-1), selects top rho * T * D positions
    
    Returns:
        Binary mask of shape (T, B, D) with 1 at selected (t,d) positions
    """
    T, B, D = exp_TBD.shape
    device = exp_TBD.device

    # compute k = rho * T * D
    rho = float(min(max(rho, 0.0), 1.0))
    k = max(1, int(round(rho * T * D)))

    # NaN-safe
    m_TBD = torch.nan_to_num(exp_TBD, nan=0.0)

    # flatten to (B, T*D)
    m_flat = m_TBD.permute(1, 0, 2).reshape(B, T * D)  # (B, T*D)

    # top-k largest per sample
    _, idx = torch.topk(m_flat, k, dim=1, largest=True)  # (B, k)

    # create mask and scatter
    mask_flat = torch.zeros((B, T * D), device=device, dtype=exp_TBD.dtype)
    mask_flat.scatter_(1, idx, 1.0)

    # reshape back to (T, B, D)
    mask_TBD = mask_flat.reshape(B, T, D).permute(1, 0, 2)

    return mask_TBD


def get_correct_classified_samples(X, times, y, gt_exps, ori_exps, predictor, Dname, batch_size=None, model_type="transformer"):
    """
    Filter samples that are correctly classified.
    
    Args:
        X: Input data [T, B, D]
        times: Time indices [T, B]
        y: Labels [B]
        gt_exps: Ground truth explanations [T, B, D]
        ori_exps: Original explanations [T, B, D]
        predictor: Classification model
        Dname: Dataset name
        batch_size: Batch size for inference
        model_type: Model type string
        
    Returns:
        Filtered tensors for correctly classified samples
    """
    T, B, D = X.shape
    predictor.eval()
    
    if batch_size is not None:
        if isinstance(predictor, CNN) or isinstance(predictor, LSTM):
            pred = torch.cat(
                [predictor(xb, tb) for xb, tb in zip(
                    torch.split(X, batch_size, dim=1), 
                    torch.split(times, batch_size, dim=1)
                )],
                dim=0
            )
        else:
            pred = torch.cat(
                [predictor(xb, tb, captum_input=False) for xb, tb in zip(
                    torch.split(X, batch_size, dim=1), 
                    torch.split(times, batch_size, dim=1)
                )],
                dim=0
            )
    else:
        if isinstance(predictor, CNN) or isinstance(predictor, LSTM):
            pred = predictor(X, times)
        else:
            pred = predictor(X, times, captum_input=False)

    correct_mask = pred.argmax(1) == y
    
    correct_X = X[:, correct_mask, :]
    correct_times = times[:, correct_mask]
    correct_labels = y[correct_mask]
    correct_gt_exps = gt_exps[:, correct_mask.cpu(), :]
    correct_ori_exps = ori_exps[:, correct_mask.cpu(), :]

    # Generate target labels
    target_labels = get_random_target_label(
        device=device, labels=correct_labels, n_classes=pred.shape[-1]
    )

    return (correct_X, correct_times, correct_labels, correct_gt_exps.float(), 
            target_labels, correct_ori_exps.float())


# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main evaluation function."""
    Dname = args.dataset.lower()

    # Load dataset
    if Dname == 'seqcombsingle':
        D = process_Synth(split_no=args.split_no, device=device, 
                         base_path=Path(args.data_path) / 'SeqCombSingle')
    elif Dname == 'scs_better':
        D = process_Synth(split_no=args.split_no, device=device, 
                         base_path=Path(args.data_path) / 'SeqCombSingleBetter')
    elif Dname == 'seqcomb_mv':
        D = process_Synth(split_no=args.split_no, device=device, 
                         base_path=Path(args.data_path) / 'SeqCombMV2/SeqCombMV2')
    elif Dname == 'lowvardetect':
        D = process_Synth(split_no=args.split_no, device=device, 
                         base_path=Path(args.data_path) / 'LowVarDetect')
    elif Dname == 'mitecg_hard':
        D = process_MITECG(split_no=args.split_no, device=device, hard_split=True, 
                          need_binarize=True, exclude_pac_pvc=True, 
                          base_path=Path(args.data_path) / 'MITECG')
    elif Dname == 'epilepsy':
        trainEpi, val, test = process_Epilepsy(split_no=args.split_no, device=device, 
                                               base_path=Path(args.data_path) / 'Epilepsy/')
    elif Dname == 'pam':
        trainEpi, val, test = process_PAM(split_no=args.split_no, device=device, 
                                          base_path=str(Path(args.data_path) / 'PAM'), 
                                          gethalf=True)

    # Process test/val data
    if Dname == 'mitecg_hard':
        _, val, test, gt_exps = D
    elif Dname in {'epilepsy', 'pam'}:
        val = (val.X, val.time, val.y)
        test = (test.X, test.time, test.y)
    else:
        test = D['test']
        val = D['val']

    # Select validation or test set
    if args.val:
        if Dname == 'mitecg_hard':
            X, times, y = val.X, val.time, val.y
        elif Dname == 'epilepsy':
            X, times, y = val
            mask = (y == 0).clone().cpu()
            X, times, y = X[:, mask, :], times[:, mask], y[mask]
        else:
            X, times, y = val
    else:
        if Dname == 'mitecg_hard':
            X, times, y = test.X, test.time, test.y
        elif Dname == 'epilepsy':
            X, times, y = test
            mask = (y == 0).clone().cpu()
            X, times, y = X[:, mask, :], times[:, mask], y[mask]
        else:
            X, times, y = test

    # Get ground truth explanations
    if Dname not in {'epilepsy', 'pam', 'mitecg_hard'} and not args.val:
        gt_exps = D['gt_exps']
    elif Dname == 'mitecg_hard' and not args.val:
        _, _, _, gt_exps = D
    else:
        # For epilepsy, pam: generate gt_exps from explainer
        if args.exp_method == 'ours':
            sdict, config = torch.load(args.model_path)
            explainer = TimeXModel(**config)
            explainer.load_state_dict(sdict)
            explainer.to(device)
            explainer.eval()
            ori_X = X.data
            out = explainer.get_saliency_explanation(ori_X, times, captum_input=False)

            T, B, d = X.shape
            gt_exps = topk_elements_TBD(
                out['mask_in'].detach().clone(),
                rho=args.real_exp_ratio,
            )
        elif args.exp_method == 'ig_batch':
            from captum.attr import IntegratedGradients
            predictor = get_model(args, X)
            predictor.load_state_dict(torch.load(args.predictor_path))
            predictor.to(device)
            predictor.eval()

            IG_batch = IntegratedGradients(predictor)
            print("Created batched IG explainer")
            
            ori_X = X.data
            print("Pre-computing IG on original inputs...")
            X_batch_orig = ori_X.permute(1, 0, 2)  # [T, B, d] -> [B, T, d]
            times_batch_orig = times.permute(1, 0)  # [T, B] -> [B, T]
            
            with torch.no_grad():
                ig_original_batch = IG_batch.attribute(
                    X_batch_orig,
                    target=y,
                    additional_forward_args=(times_batch_orig, None, True),
                    n_steps=1,
                    internal_batch_size=None
                )  # [B, T, d]
            
            ig_original = ig_original_batch.permute(1, 0, 2).float()  # [T, B, d]
            print(f"Original IG computed. Shape: {ig_original.shape}")

            gt_exps = topk_elements_TBD(
                ig_original.detach().clone(),
                rho=args.real_exp_ratio,
            )

    T, B, d = X.shape
    print(f'Dataset: {Dname}, Shape: T={T}, B={B}, d={d}')

    # Load predictor
    predictor = get_model(args, X)
    predictor.load_state_dict(torch.load(args.predictor_path))
    predictor.to(device)
    predictor.eval()

    # Load explainer and get original explanations
    ori_X = X.data

    if args.exp_method == 'ours':
        # Load TimeX/TimeX++ explainer model
        sdict, config = torch.load(args.model_path)
        explainer = TimeXModel(**config)
        explainer.load_state_dict(sdict)
        explainer.to(device)
        explainer.eval()
        
        if Dname == 'mitecg_hard':
            # For large dataset, use batch processing
            iters = torch.arange(0, B, step=1024)
            ori_exps_list = []
            for i in range(len(iters)):
                if i == (len(iters) - 1):
                    batch_X = ori_X[:, iters[i]:, :]
                    batch_times = times[:, iters[i]:]
                else:
                    batch_X = ori_X[:, iters[i]:iters[i+1], :]
                    batch_times = times[:, iters[i]:iters[i+1]]
                with torch.no_grad():
                    batch_out = explainer.get_saliency_explanation(batch_X, batch_times, captum_input=False)
                ori_exps_list.append(batch_out['mask_in'].detach().clone())
            ori_exps = torch.cat(ori_exps_list, dim=1)
        else:
            with torch.no_grad():
                ori_out = explainer.get_saliency_explanation(ori_X, times, captum_input=False)
            ori_exps = ori_out['mask_in'].detach().clone()

    elif args.exp_method == 'ig_batch':
        # Create IG object once (reuse across iterations)
        from captum.attr import IntegratedGradients
        IG_batch = IntegratedGradients(predictor)
        print("Created batched IG explainer")
        
        # Pre-compute IG on ORIGINAL input (use as target for attack)
        print("Pre-computing IG on original inputs...")
        X_batch_orig = ori_X.permute(1, 0, 2)  # [T, B, d] -> [B, T, d]
        times_batch_orig = times.permute(1, 0)  # [T, B] -> [B, T]
        
        with torch.no_grad():
            ig_original_batch = IG_batch.attribute(
                X_batch_orig,
                target=y,
                additional_forward_args=(times_batch_orig, None, True),
                n_steps=1,
                internal_batch_size=None  # Process entire batch at once
            )  # [B, T, d]
        
        ig_original = ig_original_batch.permute(1, 0, 2).float()  # [T, B, d]
        ori_exps = ig_original.detach().clone()
        explainer = None  # No explainer model for IG
        print(f"Original IG computed. Shape: {ig_original.shape}")

    else:
        # For other explainers, use gt_exps as placeholder
        explainer = None
        ori_exps = gt_exps.clone()

    print(f'The length of training dataset: {X.shape[1]}')

    # Filter correctly classified samples
    if Dname == 'mitecg_hard':
        X, times, y, gt_exps, target_labels, ori_exps = get_correct_classified_samples(
            X, times, y, gt_exps.to(device), ori_exps.to(device), predictor, Dname, 
            batch_size=1024, model_type=args.model_type
        )
    else:
        X, times, y, gt_exps, target_labels, ori_exps = get_correct_classified_samples(
            X, times, y, gt_exps.to(device), ori_exps.to(device), predictor, Dname, 
            batch_size=None, model_type=args.model_type
        )

    print(f'The length of correctly classified training dataset: {X.shape[1]}')

    T, B, d = X.shape
    print(args.exp_method, "args.exp_method", args.model_path)

    # Compute data statistics AFTER filtering (min/max along time axis T, per sample per channel)
    # Result shape: [1, B, D]
    min_X, _ = torch.min(X, dim=0, keepdim=True)  # [1, B, D]
    max_X, _ = torch.max(X, dim=0, keepdim=True)  # [1, B, D]

    # Save original data for DTW comparison
    dtw_X = X.clone()

    print(f'After filtering: T={T}, B={B}, d={d}')

    # Run attack
    if args.attack_name == 'tsef_mask_attack':
        print('Hyperparameters:')
        print("r_mt", args.tsef_attack_r_mt,
            "lr_t", args.tsef_attack_lr_t,
            "lr_f", args.tsef_attack_lr_f,
            "beta_t", args.tsef_attack_beta_t,
            "gamma_t", args.tsef_attack_gamma_t,
            "lambda_cls", args.tsef_attack_lambda_cls,
            "lambda_exp", args.tsef_attack_lambda_exp)

        transformer = get_model(args, X)
        transformer.load_state_dict(torch.load(args.predictor_path))
        transformer.to(device)
        transformer.eval()

        if args.exp_method == 'ours':
            sdict, config = torch.load(args.model_path)
            if args.org_v:
                from txai.models.modelv6_v2 import Modelv6_v2
                explainer = Modelv6_v2(**config)
            elif Dname == 'irreg':
                from txai.models.bc_model_irreg import TimeXModel_Irregular
                explainer = TimeXModel_Irregular(**config)
            else:
                explainer = TimeXModel(**config)
            explainer.load_state_dict(sdict)
            explainer.eval()
            explainer.to(device)

            if Dname == 'mitecg_hard':
                iters = torch.arange(0, X.shape[1], step=4096)
                for i in range(len(iters)):
                    if i == (len(iters) - 1):
                        batch_X = X[:, iters[i]:, :]
                        batch_gt_exps = gt_exps[:, iters[i]:, :]
                        batch_times = times[:, iters[i]:]
                        batch_target_labels = target_labels[iters[i]:]
                        batch_max_X = max_X[:, iters[i]:, :]
                        batch_min_X = min_X[:, iters[i]:, :]
                    else:
                        batch_X = X[:, iters[i]:iters[i+1], :]
                        batch_gt_exps = gt_exps[:, iters[i]:iters[i+1], :]
                        batch_times = times[:, iters[i]:iters[i+1]]
                        batch_target_labels = target_labels[iters[i]:iters[i+1]]
                        batch_max_X = max_X[:, iters[i]:iters[i+1], :]
                        batch_min_X = min_X[:, iters[i]:iters[i+1], :]

                    batch_X = tsef_mask_attack(
                        transformer, explainer, batch_X, batch_gt_exps, 
                        batch_target_labels, batch_times,
                        eps=args.attack * (batch_max_X - batch_min_X),
                        iters=args.tsef_attack_iters, min_X=batch_min_X, max_X=batch_max_X,
                        lambda_cls=args.tsef_attack_lambda_cls, 
                        lambda_exp=args.tsef_attack_lambda_exp,
                        r_mt=args.tsef_attack_r_mt,
                        k1=args.tsef_attack_k1, k2=args.tsef_attack_k2,
                        lr_t=args.tsef_attack_lr_t, lr_f=args.tsef_attack_lr_f,
                        beta_t=args.tsef_attack_beta_t, gamma_t=args.tsef_attack_gamma_t
                    )

                    if i == (len(iters) - 1):
                        X[:, iters[i]:, :] = batch_X
                    else:
                        X[:, iters[i]:iters[i+1], :] = batch_X
            else:
                X = tsef_mask_attack(
                    transformer, explainer, X, gt_exps, target_labels, times,
                    eps=args.attack * (max_X - min_X),
                    iters=args.tsef_attack_iters, min_X=min_X, max_X=max_X,
                    lambda_cls=args.tsef_attack_lambda_cls, 
                    lambda_exp=args.tsef_attack_lambda_exp,
                    r_mt=args.tsef_attack_r_mt,
                    k1=args.tsef_attack_k1, k2=args.tsef_attack_k2,
                    lr_t=args.tsef_attack_lr_t, lr_f=args.tsef_attack_lr_f,
                    beta_t=args.tsef_attack_beta_t, gamma_t=args.tsef_attack_gamma_t
                )

        elif args.exp_method in ["ig", "dyna", "ig_batch", "ig_diff"]:
            X = tsef_mask_attack_ig(
                transformer, X, gt_exps, target_labels, y, times,
                eps=args.attack * (max_X - min_X),
                iters=args.tsef_attack_iters, min_X=min_X, max_X=max_X,
                lambda_cls=args.tsef_attack_lambda_cls, 
                lambda_exp=args.tsef_attack_lambda_exp,
                r_mt=args.tsef_attack_r_mt,
                k1=args.tsef_attack_k1, k2=args.tsef_attack_k2,
                lr_t=args.tsef_attack_lr_t, lr_f=args.tsef_attack_lr_f,
                beta_t=args.tsef_attack_beta_t, gamma_t=args.tsef_attack_gamma_t
            )

    elif args.attack_name == 'clean':
        X = X.clone()
    else:
        print(f'Unknown attack: {args.attack_name}')
        raise ValueError(f"Unknown attack method: {args.attack_name}")

    # Generate explanations on adversarial examples
    if args.exp_method == 'ours':
        sdict, config = torch.load(args.model_path)
        if args.org_v:
            from txai.models.modelv6_v2 import Modelv6_v2
            model = Modelv6_v2(**config)
        elif Dname == 'irreg':
            from txai.models.bc_model_irreg import TimeXModel_Irregular
            model = TimeXModel_Irregular(**config)
        else:
            model = TimeXModel(**config)
        model.load_state_dict(sdict)
        model.eval()
        model.to(device)

        # Keep batch size at 64
        iters = torch.arange(0, B, step=64)
        generated_exps = torch.zeros_like(X)
        zx_generated = torch.zeros(X.shape[1], model.d_z)

        start_time = time.time()

        for i in range(len(iters)):
            if i == (len(iters) - 1):
                batch_X = X[:, iters[i]:, :]
                batch_times = times[:, iters[i]:]
            else:
                batch_X = X[:, iters[i]:iters[i+1], :]
                batch_times = times[:, iters[i]:iters[i+1]]

            with torch.no_grad():
                if args.savepath is None:
                    out = model.get_saliency_explanation(batch_X, batch_times, captum_input=False)
                else:
                    out = model(batch_X, batch_times, captum_input=False)

            # NOTE: below capability only works with univariate for now
            if args.org_v:
                if i == (len(iters) - 1):
                    generated_exps[:, iters[i]:, :] = torch.stack(out['mask_in'], dim=0).sum(dim=0).unsqueeze(-1).transpose(0, 1)
                else:
                    generated_exps[:, iters[i]:iters[i+1], :] = torch.stack(out['mask_in'], dim=0).sum(dim=0).unsqueeze(-1).transpose(0, 1)
            else:
                if args.savepath is None:
                    if i == (len(iters) - 1):
                        generated_exps[:, iters[i]:, :] = out['mask_in']
                    else:
                        generated_exps[:, iters[i]:iters[i+1], :] = out['mask_in']
                else:
                    if i == (len(iters) - 1):
                        generated_exps[:, iters[i]:, :] = out['mask_logits'].transpose(0, 1)
                        zx_generated[iters[i]:, :] = out['z_mask_list']
                    else:
                        generated_exps[:, iters[i]:iters[i+1], :] = out['mask_logits'].transpose(0, 1)
                        zx_generated[iters[i]:iters[i+1], :] = out['z_mask_list']

        end_time = time.time()
        print(f'Time elapsed inference TimeX, split {args.split_no}: {end_time - start_time:.6f}')

        if args.savepath is not None:
            # Get zp (prototypes)
            zp = model.prototypes.detach().clone().cpu()
            zx = zx_generated.detach().clone().cpu()
            Xsave = X.detach().clone().cpu()
            ysave = y.detach().clone().cpu()
            gexp = generated_exps.detach().clone().cpu()
            torch.save((Xsave, gexp, ysave, zx, zp), args.savepath)

    else:
        # Use other explainer APIs
        model = get_model(args, X)
        model.load_state_dict(torch.load(args.predictor_path, weights_only=False))
        model.to(device)
        model.eval()
        
        if args.model_type == "lstm" and args.exp_method in ["ig", "dyna", "ig_batch"]:
            model.train()

        explainer, needs_training = get_explainer(key=args.exp_method, args=args, device=device)
        generated_exps = torch.zeros_like(X)

        start_time = time.time()

        for i in trange(B):
            if args.exp_method == 'dyna':
                exp = explainer(model, X[:, i, :].clone(), times[:, i].clone().unsqueeze(1), 
                               y=y[i].unsqueeze(0).clone())
            else:
                exp = explainer(model, X[:, i, :].unsqueeze(1).clone(), 
                               times[:, i].unsqueeze(-1).clone(), y[i].unsqueeze(0).clone())

            if exp.shape[0] == 1:
                exp = exp.squeeze(0)

            if exp.shape != (generated_exps.shape[0], generated_exps.shape[2]):
                raise ValueError(f"Explanation shape {exp.shape} doesn't match expected")

            generated_exps[:, i, :] = exp

        end_time = time.time()
        print(f'Time elapsed inference {args.exp_method}, split {args.split_no}: {end_time - start_time:.6f}')

    # Evaluate results
    results_dict = ground_truth_xai_eval(generated_exps, gt_exps)
    iou_dict = ground_truth_IoU(generated_exps, gt_exps)
    
    results_dict.update(iou_dict)

    # Predictor evaluation
    predictor = get_model(args, X)
    predictor.load_state_dict(torch.load(args.predictor_path))
    predictor.to(device)
    predictor.eval()

    bs = 16 if Dname == 'mitecg_hard' else 32
    test_perturb = (X, times, target_labels, y)

    if Dname == 'epilepsy':
        f1, pre_asr = eval_mvts_transformer_asr(test_perturb, predictor, batch_size=bs, 
                                                auroc=False, auprc=False)
        results_dict['predictor_f1_target'] = [f1]
        results_dict['predictor_asr'] = [pre_asr]
    else:
        f1, pre_auprc, pre_auroc, pre_asr = eval_mvts_transformer_asr(
            test_perturb, predictor, batch_size=bs, auroc=True, auprc=True
        )
        results_dict['predictor_f1_target'] = [f1]
        results_dict['predictor_auprc_target'] = [pre_auprc]
        results_dict['predictor_auroc_target'] = [pre_auroc]
        results_dict['predictor_asr'] = [pre_asr]

    # Print results
    print(f'Results for {args.exp_method} explainer on {args.dataset} with split={args.split_no}')
    for k, v in results_dict.items():
        print(f'\t{k} \t = {np.mean(v):.4f} +- {np.std(v) / np.sqrt(len(v)):.4f}')

    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSEF Attack Evaluation')
    
    # Basic arguments
    parser.add_argument('--exp_method', type=str, default='ours', 
                       help="Explainer method: ['ours', 'ig', 'dyna', 'ig_batch', 'ig_diff']")
    parser.add_argument('--dataset', type=str, default="lowvardetect",
                       help="Dataset name")
    parser.add_argument('--split_no', default=1, type=int,
                       help="Data split number (use -1 for all splits)")
    parser.add_argument('--model_path', type=str, help='Path to explainer model')
    parser.add_argument('--attack_name', type=str, default='tsef_mask_attack',
                       help="Attack method: ['tsef_mask_attack', 'clean']")
    parser.add_argument('--model_type', type=str, default="transformer", 
                       choices=["transformer", "cnn", "lstm"])
    parser.add_argument('--org_v', action='store_true')
    parser.add_argument('--data_path', default="./datasets/", type=str,
                       help='Path to datasets root')
    parser.add_argument('--results_path', default="./results/", type=str,
                       help='Path to save attack results')
    parser.add_argument('--savepath', default=None, type=str)
    parser.add_argument('--save_attack_name', default=None, type=str)
    parser.add_argument('--predictor_path', type=str, help='Path to classifier model')
    parser.add_argument('--is_timex', action='store_true', default=False,
                       help='Use TimeX model (bc_model4) if True, otherwise use bc_model')
    parser.add_argument('--val', action='store_true', default=False,
                       help='Use validation set instead of test set')
    parser.add_argument('--attack', default=0.1, type=float,
                       help='Attack strength (Attack budget, e.g., 0.1 = 10%)')

    # TSEF attack hyperparameters
    parser.add_argument('--tsef_attack_r_mt', default=0.5, type=float,
                       help='Sparsity ratio for temporal mask')
    parser.add_argument('--tsef_attack_iters', default=100, type=int,
                       help='Number of outer iterations')
    parser.add_argument('--tsef_attack_k1', default=10, type=int,
                       help='Inner iterations for Mt optimization')
    parser.add_argument('--tsef_attack_k2', default=10, type=int,
                       help='Inner iterations for Mf optimization')
    parser.add_argument('--tsef_attack_lr_t', default=1.0, type=float,
                       help='Learning rate for temporal mask')
    parser.add_argument('--tsef_attack_lr_f', default=1.0, type=float,
                       help='Learning rate for frequency mask')
    parser.add_argument('--tsef_attack_beta_t', default=2.0, type=float,
                       help='Weight for temporal mask sparsity loss')
    parser.add_argument('--tsef_attack_gamma_t', default=1.0, type=float,
                       help='Weight for connectivity loss')
    parser.add_argument('--tsef_attack_lambda_cls', default=1.0, type=float,
                       help='Weight for classification loss')
    parser.add_argument('--tsef_attack_lambda_exp', default=1.0, type=float,
                       help='Weight for explanation loss')
    parser.add_argument('--real_exp_ratio', default=0.3, type=float,
                       help='Ratio of top elements to select for real dataset explanations')

    args = parser.parse_args()
    
    # Select which TimeXModel to use based on command-line argument
    if args.is_timex:
        from txai.models.bc_model4 import TimeXModel
        print("Using TimeX Model")
    else:
        from txai.models.bc_model import TimeXModel
        print("Using TimeX++ Model")
    
    # Build save_name with hyperparameters
    base_name = args.save_attack_name or args.attack_name
    if args.dataset.lower() in ['pam', 'epilepsy']:
        save_name = f"{base_name}_lambda_cls_{args.tsef_attack_lambda_cls}_lambda_exp_{args.tsef_attack_lambda_exp}_real_exp_ratio_{args.real_exp_ratio}"
    else:
        save_name = f"{base_name}_lambda_cls_{args.tsef_attack_lambda_cls}_lambda_exp_{args.tsef_attack_lambda_exp}"
    
    if args.split_no == -1:
        # Run on all splits
        results_attack = {attack: {} for attack in [args.attack]}
        for attack in [args.attack]:
            results = {}
            for split in range(1, 6):
                if args.model_path is not None:
                    args.model_path = re.sub(r"split=\d", f"split={split}", args.model_path)
                if args.predictor_path is not None:
                    args.predictor_path = re.sub(r"split=\d", f"split={split}", args.predictor_path)
                args.attack = attack
                print(f"model path: {args.model_path}")
                print(f"predictor path: {args.predictor_path}")
                args.split_no = split
                split_results = main(args)
                for k, v in split_results.items():
                    if k not in results:
                        results[k] = []
                    results[k].extend(v)
            
            print(f'Results for {args.exp_method} explainer on all splits')
            for k, v in results.items():
                print(f'\t{k} \t = {np.mean(v):.4f} +- {np.std(v) / np.sqrt(len(v)):.4f}')
                results_attack[attack][k] = [np.mean(v), np.std(v) / np.sqrt(len(v))]

            # Save results for this attack
            os.makedirs(args.results_path, exist_ok=True)
            save_path = os.path.join(args.results_path, f'attack_results_{save_name}_attack_{attack}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
        
        # Print summary
        print(f'Results for {args.exp_method} explainer on all {args.attack_name} attacks')
        for k, v in results_attack.items():
            for metric, value in v.items():
                print(f'\tattack {k} \t \t{metric} \t = {value[0]:.4f} +- {value[1]:.4f}')
        
        # Save results_attack summary
        save_path = os.path.join(args.results_path, f'attack_results_{save_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(results_attack, f)
    else:
        # Run on single split
        results_attack = {attack: {} for attack in [args.attack]}
        for attack in [args.attack]:
            results = {}
            if args.model_path is not None:
                args.model_path = re.sub(r"split=\d", f"split={args.split_no}", args.model_path)
            if args.predictor_path is not None:
                args.predictor_path = re.sub(r"split=\d", f"split={args.split_no}", args.predictor_path)
            args.attack = attack
            print(f"model path: {args.model_path}")
            print(f"predictor path: {args.predictor_path}")
            
            split_results = main(args)
            for k, v in split_results.items():
                if k not in results:
                    results[k] = []
                results[k].extend(v)
            
            print(f'Results for {args.exp_method} explainer on split {args.split_no}')
            for k, v in results.items():
                print(f'\t{k} \t = {np.mean(v):.4f} +- {np.std(v) / np.sqrt(len(v)):.4f}')
                results_attack[attack][k] = [np.mean(v), np.std(v) / np.sqrt(len(v))]

            # Save results for this attack and split
            os.makedirs(args.results_path, exist_ok=True)
            save_path = os.path.join(args.results_path, f'attack_results_{save_name}_attack_{attack}_split_{args.split_no}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
        
        # Print summary
        print(f'Results for {args.exp_method} explainer on all {args.attack_name} attacks')
        for k, v in results_attack.items():
            for metric, value in v.items():
                print(f'\tattack {k} \t \t{metric} \t = {value[0]:.4f} +- {value[1]:.4f}')
        
        # Save results_attack summary
        save_path = os.path.join(args.results_path, f'attack_results_{save_name}_split_{args.split_no}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(results_attack, f)
