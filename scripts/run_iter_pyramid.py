"""
Iteration function for networks with motion pyramid and warping.
"""
import sys
import time

sys.path.append('../')
from itertools import combinations, permutations
from random import choice, sample, shuffle
from typing import (Callable, Dict, List, Literal, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from mmengine import Config, ConfigDict
from monai.data import DataLoader
from monai.networks.utils import one_hot
from torch.optim import Optimizer
from utils import optional_context

CFG = Union[dict, Config, ConfigDict]


def run_iter(cfg: CFG,
             model: nn.Module,
             reg_head: nn.Module,
             reg_head_ms: Sequence[nn.Module],
             dataloaders: Sequence[DataLoader],
             loss_funcs: Sequence[Callable],
             metric_funcs: Sequence[Callable],
             scaler: torch.cuda.amp.GradScaler,
             optimizer: Optimizer,
             epoch_iter: int,
             max_len: int,
             phase: str):
    """
    Run one iteration of training or validation. It handles combining dataloaders, losses computation, logging and metric calculation.

    Args:
        cfg: (CFG) Configuration object.
        model: (nn.Module) Model to be trained.
        reg_head: (nn.Module) Registration head for upsampling / velocity integration and image warping.
        reg_head_ms: (Sequence[nn.Module]) List of registration heads for multi-level registration.
        dataloaders: (Sequence[DataLoader]) List of dataloaders, from different datasets, of different lengths.
        loss_funcs: (Sequence[Callable]) Dictionary of loss functions.
        metric_funcs: (Sequence[Callable]) Dictionary of metric functions.
        scaler: (torch.cuda.amp.GradScaler) Gradient scaler for automatic mixed precision training.
        optimizer: (Optimizer) Optimizer for updating the model parameters.
        epoch_iter: (int) Current iteration number.
        max_len: (int) Maximum length of the dataloaders.
        phase: (str) Phase of the iteration, either 'train' or 'val'.
    """
    logging_dict = dict()

    # combine dataloaders with different sizes
    idxs = list(range(len(dataloaders)))
    dataiters = [iter(d) for d in dataloaders]
    k = 0

    for _ in range(max_len):
        shuffle(idxs)
        for i in idxs:

            try:
                data = next(dataiters[i])
            except StopIteration:
                continue

            # log number of iteration
            logging_dict.update({'iter': epoch_iter + k})
            k += 1

            # retrieve images and one-hot label maps
            source = data['image'][:cfg.batch_size // 2].float().to(cfg.device)
            source_oh = data['label'][:cfg.batch_size // 2].float().to(cfg.device)
            target = data['image'][cfg.batch_size // 2:].float().to(cfg.device)
            target_oh = data['label'][cfg.batch_size // 2:].float().to(cfg.device)

            total_loss = 0.0
            with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=cfg.use_amp):
                fwd_flows, _ = model(source, target)

                final_fwd_flow, y_source, y_source_oh = reg_head(fwd_flows[-1], source, source_oh)

                ################################ loss at top level ################################
                with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=False):
                    # amp has problem when computing LNCC similarity
                    # image similarity
                    fwd_sim = loss_funcs['sim'](y_source, target)
                #
                total_loss += cfg.sim_loss_cfg.weight * fwd_sim
                #
                logging_dict.update({'fwd_sim': fwd_sim.detach().cpu()})

                # regularization loss at top level
                reg = loss_funcs['reg'](fwd_flows[-1])
                #
                total_loss += cfg.reg_loss_cfg.weight * reg
                #
                logging_dict.update({'reg': reg.detach().cpu()})

                with optional_context(cfg.dice_loss_cfg.weight == 0.0, torch.no_grad()):
                    # dice loss
                    dice = loss_funcs['dice'](y_source_oh, target_oh)
                    #
                    total_loss += cfg.dice_loss_cfg.weight * dice
                    #
                    logging_dict.update({'dice': dice.detach().cpu()})

                ################################ loss at multi-level ################################
                # multi-scale registration
                for i, (scale, scale_loss_weight) in enumerate(zip(cfg.scale_pyramid, cfg.scale_loss_weights)):
                    source_down = F.avg_pool3d(
                        source,
                        kernel_size=scale,
                        stride=scale
                    )
                    target_down = F.avg_pool3d(
                        target,
                        kernel_size=scale,
                        stride=scale
                    )
                    y_source_down, _ = reg_head_ms[i](fwd_flows[i], source_down)

                    with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=False):
                        fwd_sim_down = loss_funcs['sim'](y_source_down, target_down)
                    #
                    total_loss += scale_loss_weight * cfg.sim_loss_cfg.weight * fwd_sim_down
                    #
                    logging_dict.update({f'fwd_sim_{scale}': fwd_sim_down.detach().cpu()})

                    #
                    reg_down = loss_funcs['reg'](fwd_flows[i])
                    #
                    total_loss += scale_loss_weight * cfg.reg_loss_cfg.weight * reg_down
                    #
                    logging_dict.update({f'reg_{scale}': reg_down.detach().cpu()})

                #
                logging_dict.update({'total_loss': total_loss.detach().cpu()})

                if phase == 'train':
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            with torch.no_grad():
                # dice
                init_dice = metric_funcs['dice'](source_oh, target_oh)

                # compute foreground binary mask
                target_fg = torch.where(target > 0.0, 1.0, 0.0).float().detach().to(cfg.device)
                # source_fg = torch.where(source > 0.0, 1.0, 0.0).float().detach().to(cfg.device)

                # Jac det
                # since the flow is mapping from target space to source space, we compute the JacDet within the target foreground
                log_jacdet, non_pos_jacdet = metric_funcs['jacdet'](
                    final_fwd_flow.detach().cpu().float().numpy(), target_fg.detach().cpu().numpy())

                logging_dict.update({
                    'init_dice':
                        init_dice.detach().mean().cpu().item(),
                    'log_jacdet':
                        log_jacdet.mean(),
                    'non_pos_jacdet':
                        non_pos_jacdet.mean(),
                })

            # wandb logging
            wandb.log({phase: logging_dict})
