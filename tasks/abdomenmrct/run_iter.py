import os
import sys
import time
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from itertools import combinations, permutations
from random import choice, sample
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
from utils import optional_context, iterate_paired_dataloaders
from prealign import PreAlign

CFG = Union[dict, Config, ConfigDict]


def run_iter(cfg: CFG,
             model: nn.Module,
             reg_head: Sequence[nn.Module],
             dataloaders: Sequence[DataLoader],
             prealigner: PreAlign,
             loss_funcs: Dict[str, Callable],
             metric_funcs: Dict[str, Callable],
             scaler: torch.cuda.amp.GradScaler,
             optimizer: Optimizer,
             current_iter: int,
             phase: str) -> int:
    logging_dict = dict()
    ct_dataloader, mr_dataloader = dataloaders
    total_pairs = 0
    valid_pairs = 0

    for i, (ct_data, mr_data) in enumerate(
            iterate_paired_dataloaders(ct_dataloader, mr_dataloader, reset_shorter=True, max_samples=cfg.max_samples)):
        total_pairs += 1
        # Process the pair through the workflow
        # Note: labels are already one-hot encoded from the dataloader
        processed = prealigner.process_pair(
            src_img=mr_data['image'],
            src_oh=mr_data['label'],  # already one-hot encoded
            tgt_img=ct_data['image'],
            tgt_oh=ct_data['label']   # already one-hot encoded
        )

        # Extract processed data
        source = processed['src_image'].float().to(cfg.device)
        target = processed['tgt_image'].float().to(cfg.device)
        src_oh = processed['src_seg'].float().to(cfg.device)  # already one-hot encoded
        tgt_oh = processed['tgt_seg'].float().to(cfg.device)  # already one-hot encoded

        # Compute initial Dice score before model forward
        with torch.no_grad():
            init_dice = metric_funcs['dice'](src_oh, tgt_oh)
            init_dice_mean = init_dice.detach().mean().cpu().item()

            # Skip this pair if initial Dice is NaN or below threshold
            if torch.isnan(init_dice).any() or np.isnan(init_dice_mean):
                print(f"Skipping pair {i} due to NaN initial Dice score")
                continue

            # Skip if initial Dice is below threshold (default to 0.0 if not specified)
            dice_threshold = getattr(cfg, 'init_dice_threshold', 0.0)
            if init_dice_mean < dice_threshold:
                print(f"Skipping pair {i} due to low initial Dice score: {init_dice_mean:.4f} < {dice_threshold:.4f}")
                continue

        valid_pairs += 1
        # Only update iteration counter and logging dict if we're actually processing this pair
        current_iter += 1
        logging_dict.update({'iter': current_iter})

        # Update logging dict with initial Dice
        logging_dict.update({'init_dice': init_dice_mean})

        with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=cfg.use_amp):
            total_loss = 0.0
            if not cfg.pyramid:
                if 'Dual' in cfg.model_cfg.type and cfg.model_cfg.bidirectional:
                    fwd_flow, bck_flow = model(source, target)
                else:
                    fwd_flow = model(source, target)
                    bck_flow = model(target, source)
            else:
                fwd_py_flows, bck_py_flows = model(source, target)  # bidirectional = True
                fwd_flow = fwd_py_flows[-1]
                bck_flow = bck_py_flows[-1]

            flow = torch.cat([fwd_flow, bck_flow], dim=0).to(cfg.device)

            fwd_flow, y_source, y_source_oh = reg_head[-1](fwd_flow, source, src_oh)
            bck_flow, y_target, y_target_oh = reg_head[-1](bck_flow, target, tgt_oh)

            with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=False):
                # image similarity
                fwd_sim = loss_funcs['sim'](y_source, target)
                bck_sim = loss_funcs['sim'](y_target, source)
            #
            total_loss += cfg.sim_loss_cfg.weight * 0.5 * (fwd_sim + bck_sim)
            #
            logging_dict.update({'fwd_sim': fwd_sim.detach().cpu()})
            logging_dict.update({'bck_sim': bck_sim.detach().cpu()})

            # regularization
            reg = loss_funcs['reg'](flow)
            #
            total_loss += cfg.reg_loss_cfg.weight * reg
            #
            logging_dict.update({'reg': reg.detach().cpu()})

            # multi-scale registration
            if cfg.pyramid:
                for k, (scale, scale_loss_weight) in enumerate(zip(cfg.scale_pyramid, cfg.scale_loss_weights)):
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
                    #
                    y_source_down, _ = reg_head[k](fwd_py_flows[k], source_down)
                    y_target_down, _ = reg_head[k](bck_py_flows[k], target_down)
                    #
                    with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=False):
                        fwd_sim_down = loss_funcs['sim'](y_source_down, target_down)
                        bck_sim_down = loss_funcs['sim'](y_target_down, source_down)
                    #
                    total_loss += scale_loss_weight * cfg.sim_loss_cfg.weight * 0.5 * (fwd_sim_down + bck_sim_down)
                    #
                    logging_dict.update({f'fwd_sim_{scale}': fwd_sim_down.detach().cpu(),
                                         f'bck_sim_{scale}': bck_sim_down.detach().cpu()})
                    #
                    reg_down = loss_funcs['reg'](torch.cat([fwd_py_flows[k], bck_py_flows[k]], dim=0))
                    #
                    total_loss += scale_loss_weight * cfg.reg_loss_cfg.weight * reg_down
                    #
                    logging_dict.update({f'reg_{scale}': reg_down.detach().cpu()})

            with optional_context(cfg.dice_loss_cfg.weight == 0.0, torch.no_grad()):
                # dice loss
                fwd_dice = loss_funcs['dice'](y_source_oh, tgt_oh)
                bck_dice = loss_funcs['dice'](y_target_oh, src_oh)
                #
                total_loss += cfg.dice_loss_cfg.weight * 0.5 * (fwd_dice + bck_dice)
                #
                logging_dict.update({'fwd_dice': fwd_dice.detach().cpu()})
                logging_dict.update({'bck_dice': bck_dice.detach().cpu()})


            #
            logging_dict.update({'total_loss': total_loss.detach().cpu()})

            if phase == 'train':
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            # Jac det
            log_jacdet, non_pos_jacdet = metric_funcs['jacdet'](
                fwd_flow.detach().cpu().float().numpy())

            logging_dict.update({
                'log_jacdet': log_jacdet.mean(),
                'non_pos_jacdet': non_pos_jacdet.mean(),
            })

            # wandb logging
            wandb.log({phase: logging_dict})

    # Log pair statistics at the end of iteration
    pair_stats = {
        f'{phase}/total_pairs': total_pairs,
        f'{phase}/valid_pairs': valid_pairs,
        f'{phase}/valid_pair_ratio': valid_pairs / total_pairs if total_pairs > 0 else 0
    }
    wandb.log(pair_stats)

    return current_iter
