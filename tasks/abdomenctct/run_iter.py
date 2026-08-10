import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from typing import Callable, Dict, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from mmengine import Config, ConfigDict
from monai.data import DataLoader
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
             epoch_iter: int,
             phase: str):
    logging_dict = dict()
    for i, data in enumerate(dataloader):
        logging_dict.update({'iter': epoch_iter + i})

        if np.random.uniform() > 0.5:
            source = data['fixed']
            target = data['moving']
            src_oh = data['fixed_label']
            tgt_oh = data['moving_label']
        else:
            source = data['moving']
            target = data['fixed']
            src_oh = data['moving_label']
            tgt_oh = data['fixed_label']

        source = source.float().to(cfg.device)
        target = target.float().to(cfg.device)
        src_oh = src_oh.float().to(cfg.device)
        tgt_oh = tgt_oh.float().to(cfg.device)

        if np.random.uniform() > 0.5:
            # flip 'RL', [B, C, H, W, D]
            source = source.flip(dims=[2])
            target = target.flip(dims=[2])
            src_oh = src_oh.flip(dims=[2])
            tgt_oh = tgt_oh.flip(dims=[2])

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
                    source_down = F.avg_pool3d(source, kernel_size=scale, stride=scale)
                    target_down = F.avg_pool3d(target, kernel_size=scale, stride=scale)
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

            # dice loss (auxiliary, weight may be 0)
            with optional_context(cfg.dice_loss_cfg.weight == 0.0, torch.no_grad()):
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

            jacdet_voxels = np.prod(np.maximum(np.array(fwd_flow.shape[2:]) - 4, 1))
            logging_dict.update({
                'log_jacdet': log_jacdet.mean(),
                'np_jacdet': non_pos_jacdet.mean() / jacdet_voxels * 100,
            })

            # wandb logging
            wandb.log({phase: logging_dict})
