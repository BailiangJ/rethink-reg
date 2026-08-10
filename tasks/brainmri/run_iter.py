import os
import sys
import time
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
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
             reg_head: Sequence[nn.Module],
             dataloader: DataLoader,
             loss_funcs: Dict[str, Callable],
             metric_funcs: Dict[str, Callable],
             scaler: torch.cuda.amp.GradScaler,
             optimizer: Optimizer,
             current_iter: int,
             phase: str):
    logging_dict = dict()
    for i, data in enumerate(dataloader):
        logging_dict.update({'iter': current_iter + i})

        source = data['image'][:cfg.batch_size // 2].float().to(cfg.device)
        target = data['image'][cfg.batch_size // 2:].float().to(cfg.device)

        flip_aug = np.random.uniform() > 0.5
        if flip_aug:
            source = source.flip(dims=[2])
            target = target.flip(dims=[2])

        target_fg = torch.where(target > 0.0, 1.0, 0.0).float().detach().to(cfg.device)
        # source_fg = torch.where(source > 0.0, 1.0, 0.0).float().detach().to(cfg.device)

        with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=cfg.use_amp):
            total_loss = 0.0

            if not cfg.pyramid:
                if 'Dual' in cfg.model_cfg.type and cfg.model_cfg.get('bidirectional', cfg.model_cfg.get('config', {}).get('bidirectional', False)):
                    fwd_flow, bck_flow = model(source, target)
                else:
                    fwd_flow = model(source, target)
                    bck_flow = model(target, source)
            else:
                fwd_py_flows, bck_py_flows = model(source, target)  # bidirectional = True
                fwd_flow = fwd_py_flows[-1]
                bck_flow = bck_py_flows[-1]

            flow = torch.cat([fwd_flow, bck_flow], dim=0).to(cfg.device)

            fwd_flow, y_source, _ = reg_head[-1](fwd_flow, source)
            bck_flow, y_target, _ = reg_head[-1](bck_flow, target)

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

            with optional_context(cfg.np_jacdet_weight==0.0, torch.no_grad()):
                np_jacdet = loss_funcs['jacdet'](torch.cat([fwd_flow, bck_flow], dim=0))
                #
                total_loss += cfg.np_jacdet_weight * np_jacdet
                #
                logging_dict.update({'np_jacdet':np_jacdet.detach().cpu().item()})


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
                fwd_flow.detach().cpu().float().numpy(), target_fg.detach().cpu().numpy())

            logging_dict.update({
                'log_jacdet':
                    log_jacdet.mean(),
            })

            # log VFA beta parameters per level
            if 'VFA' in cfg.model_cfg.type and hasattr(model, 'decoder') and hasattr(model.decoder, 'decoder'):
                for idx, vfa_block in enumerate(model.decoder.decoder):
                    if hasattr(vfa_block, 'beta'):
                        scale = 2 ** model.decoder.out_indices[idx]
                        logging_dict[f'beta_{scale}'] = vfa_block.beta.item()

            # wandb logging
            wandb.log({phase: logging_dict})
