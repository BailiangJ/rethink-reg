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
             dataloaders: Sequence[DataLoader],
             metric_funcs: Dict[str, Callable],
             current_iter: int,
             phase: str) -> int:
    batch_size = 2
    logging_dict = dict()
    for i, data in enumerate(dataloaders):
        logging_dict.update({'iter': current_iter + i})

        source = data['image'][:batch_size // 2].float().to(cfg.device)
        target = data['image'][batch_size // 2:].float().to(cfg.device)
        src_oh = one_hot(data['label'][:batch_size//2], num_classes=10).float().to(cfg.device)
        tgt_oh = one_hot(data['label'][batch_size // 2:], num_classes=10).float().to(cfg.device)

        with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=cfg.use_amp):
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

            fwd_flow, y_source, y_src_oh = reg_head[-1](fwd_flow, source, src_oh)
            bck_flow, y_target, y_tgt_oh = reg_head[-1](bck_flow, target, tgt_oh)

            flow = torch.cat([fwd_flow, bck_flow], dim=0).to(cfg.device)

        with torch.no_grad():
            # include_background=False, get_not_nans=True
            fwd_dice,_ = metric_funcs['dice'](y_src_oh, tgt_oh)
            bck_dice,_ = metric_funcs['dice'](y_tgt_oh, src_oh)
            # Jac det
            log_jacdet, non_pos_jacdet = metric_funcs['jacdet'](
                flow.detach().cpu().float().numpy())

            logging_dict.update({
                'fwd_dice':
                fwd_dice.detach().mean().cpu().item(),
                'bck_dice':
                bck_dice.detach().mean().cpu().item(),
                'log_jacdet':
                    log_jacdet.mean(),
                'non_pos_jacdet':
                    non_pos_jacdet.mean(),
            })

            # wandb logging
            wandb.log({phase: logging_dict})
    return current_iter+i+1
