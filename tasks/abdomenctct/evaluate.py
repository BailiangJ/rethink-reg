"""Evaluate abdomen CT-CT registration on PSMAReg.

PSMAReg pairs are longitudinal (baseline / follow-up) scans of the same patient,
affine-aligned offline, so this evaluator is inter-subject only in the sense that
anatomy has changed between visits. Metrics are reported per organ, and organs
missing from either scan of a pair are excluded rather than scored as zero.

The architecture is not re-specified here: train.py dumps the resolved config to
<method_folder>/exp<N>/train_configs.py, and this script reads the model_cfg, the
registration head and the data settings back from it.

Examples:
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 100
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 100 --split val
    python evaluate.py -m ./pwc_iter_outputs -exp 0 -epoch 100 --flow-idx 0 --scale 16
"""
import json
import os
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
import torch
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import compute_hausdorff_distance, compute_surface_dice
from utils import (load_data_PSMAReg_pair, worker_init_fn, get_PSMAReg_organ_names,
                   resolve_path)
from models import build_metrics, build_registration_head, build_flow_estimator
from models.metrics import calc_jac_dets, calc_measurements, get_identity_grid

torch.backends.cudnn.deterministic = True


def compute_dice_per_channel(pred, target):
    spatial_dims = tuple(range(2, pred.ndim))
    intersection = (pred * target).sum(dim=spatial_dims)
    denominator = pred.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
    return (2.0 * intersection / denominator.clamp_min(1e-8)).squeeze(0).cpu().numpy()


def compute_hd95_per_channel(pred, target, valid, compute_hd95):
    hd95 = np.full(pred.shape[1], np.nan, dtype=np.float32)
    pred_present = pred.flatten(2).sum(dim=2).squeeze(0) > 0
    target_present = target.flatten(2).sum(dim=2).squeeze(0) > 0
    valid_indices = [i for i, is_valid in enumerate(valid)
                     if is_valid and pred_present[i].item() and target_present[i].item()]
    if not valid_indices:
        return hd95

    idx = torch.as_tensor(valid_indices, device=pred.device)
    values = compute_hd95(pred.index_select(1, idx), target.index_select(1, idx)).reshape(-1)
    for i, value in zip(valid_indices, values.detach().cpu().numpy()):
        hd95[i] = value
    return hd95


def get_nsd_spacing(cfg):
    spacing = cfg.get('spacing', None)
    if spacing is not None:
        return tuple(float(x) for x in spacing)

    dataset_json = os.path.join(resolve_path(cfg.data_dir), 'PSMAReg_dataset.json')
    if os.path.exists(dataset_json):
        with open(dataset_json, 'r') as f:
            spacing = json.load(f).get('target_spacing')
        if spacing is not None:
            return tuple(float(x) for x in spacing)

    return (2.7344, 2.7344, 3.27)


def compute_nsd_per_channel(pred, target, valid, compute_nsd, class_thresholds):
    nsd = np.full(pred.shape[1], np.nan, dtype=np.float32)
    pred_present = pred.flatten(2).sum(dim=2).squeeze(0) > 0
    target_present = target.flatten(2).sum(dim=2).squeeze(0) > 0
    valid_indices = [i for i, is_valid in enumerate(valid)
                     if is_valid and pred_present[i].item() and target_present[i].item()]
    if not valid_indices:
        return nsd

    idx = torch.as_tensor(valid_indices, device=pred.device)
    thresholds = [class_thresholds[i] for i in valid_indices]
    values = compute_nsd(pred.index_select(1, idx),
                         target.index_select(1, idx),
                         thresholds).reshape(-1)
    for i, value in zip(valid_indices, values.detach().cpu().numpy()):
        nsd[i] = value
    return nsd


def compute_ndv(flow, foreground_mask=None):
    if flow.shape[0] != 1:
        raise ValueError(f'compute_ndv expects batch size 1, got {flow.shape[0]}')

    disp_field = flow.squeeze(0).detach().cpu().float().numpy()
    def_field = disp_field + get_identity_grid(disp_field)
    jac_dets = calc_jac_dets(def_field)

    if foreground_mask is None:
        mask = np.ones_like(def_field[0, 1:-1, 1:-1, 1:-1], dtype=np.float32)
    else:
        if torch.is_tensor(foreground_mask):
            mask = foreground_mask.detach().cpu().float().numpy()
        else:
            mask = np.asarray(foreground_mask, dtype=np.float32)
        mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.float32)
        mask = mask[1:-1, 1:-1, 1:-1]

    _, _, ndv, _ = calc_measurements(jac_dets, mask)
    total_voxels = np.sum(mask)
    per_ndv = ndv / total_voxels if total_voxels > 0 else np.nan
    return float(ndv), float(per_ndv)


def non_pos_jacdet_percent(non_pos_jacdet, flow):
    jacdet_voxels = np.prod(np.maximum(np.array(flow.shape[2:]) - 4, 1))
    return non_pos_jacdet.mean() / jacdet_voxels * 100


def get_fwd_flows(model, source, target):
    """Every forward flow the model produces, coarse to fine.

    Iterative models expose forward_eval(), which additionally returns the
    intermediate flow of each refinement iteration, not just the last one per
    level. Single-flow models return one tensor, wrapped here into a list so
    that flow_idx=-1 uniformly selects the model's actual output.
    """
    if hasattr(model, 'forward_eval'):
        flows = model.forward_eval(source, target)['fwd']
    else:
        flows = model(source, target)
        # bidirectional models return (fwd, bck); only the forward flow is scored
        if isinstance(flows, tuple):
            flows = flows[0]

    if not isinstance(flows, (list, tuple)):
        return [flows]
    if not flows:
        raise ValueError('the model returned no forward flow.')
    return list(flows)


def eval(cfg):
    print(cfg.load_model)
    model = build_flow_estimator(cfg.model_cfg)
    model.load_state_dict(
        torch.load(cfg.load_model, map_location=torch.device(cfg.device)))
    model.to(cfg.device)
    model.eval()

    # build registration head module. The default cfg.registration_cfg upsamples
    # by 2 (the final flow is at half resolution); scoring a coarser pyramid
    # level needs the matching factor, e.g. --scale 16 with --flow-idx 0.
    registration_cfg = cfg.registration_cfg
    if cfg.scale is not None:
        registration_cfg = dict(type='RegistrationHead',
                                image_size=cfg.image_size,
                                spatial_scale=cfg.scale,
                                flow_scale=cfg.scale,
                                interp_mode='bilinear')
    reg_head = build_registration_head(registration_cfg)
    reg_head.to(cfg.device)
    reg_head.eval()

    label_registration_cfg = deepcopy(registration_cfg)
    label_registration_cfg.update(dict(interp_mode='nearest'))
    label_reg_head = build_registration_head(label_registration_cfg)
    label_reg_head.to(cfg.device)
    label_reg_head.eval()

    eval_split = cfg.get('eval_split', 'test')
    dataset = load_data_PSMAReg_pair(cfg, split=eval_split,
                                     cache_rate=cfg.cache_rate,
                                     num_workers=cfg.num_workers)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.get('dataloader_num_workers', 0),
                            worker_init_fn=worker_init_fn)

    compute_hd95 = partial(compute_hausdorff_distance, include_background=True, percentile=95)
    nsd_spacing = get_nsd_spacing(cfg)
    print(f'NSD spacing: {nsd_spacing}')
    compute_nsd = partial(compute_surface_dice,
                          include_background=True,
                          spacing=nsd_spacing,
                          use_subvoxels=False)
    compute_jacdet = build_metrics(dict(type='sdlogjac'))

    organ_names = get_PSMAReg_organ_names(cfg.label_list)
    assert len(organ_names) == len(cfg.label_list)

    metrics_list = []

    with torch.no_grad():
        for data in dataloader:
            source = data['moving'].float().to(cfg.device)
            target = data['fixed'].float().to(cfg.device)
            src_oh = data['moving_label'].float().to(cfg.device)
            tgt_oh = data['fixed_label'].float().to(cfg.device)

            fwd_flows = get_fwd_flows(model, source, target)
            if not -len(fwd_flows) <= cfg.flow_idx < len(fwd_flows):
                raise IndexError(f'--flow-idx {cfg.flow_idx} is out of range '
                                 f'for {len(fwd_flows)} flows.')
            fwd_flow = fwd_flows[cfg.flow_idx]

            flow, y_source, _ = reg_head(fwd_flow, source, None)
            _, _, y_source_oh = label_reg_head(fwd_flow, None, src_oh)
            y_source_oh = (y_source_oh > 0.5).float()

            pair_valid = ((src_oh.flatten(2).sum(dim=2) > 0) &
                          (tgt_oh.flatten(2).sum(dim=2) > 0)).squeeze(0).cpu().numpy()

            init_dice = compute_dice_per_channel(src_oh, tgt_oh)
            reg_dice = compute_dice_per_channel(y_source_oh, tgt_oh)
            init_dice[~pair_valid] = np.nan
            reg_dice[~pair_valid] = np.nan

            init_hd95 = compute_hd95_per_channel(src_oh, tgt_oh, pair_valid, compute_hd95)
            reg_hd95 = compute_hd95_per_channel(y_source_oh, tgt_oh, pair_valid, compute_hd95)

            class_thresholds = [1] * src_oh.shape[1]
            init_nsd = compute_nsd_per_channel(src_oh, tgt_oh, pair_valid, compute_nsd, class_thresholds)
            reg_nsd = compute_nsd_per_channel(y_source_oh, tgt_oh, pair_valid, compute_nsd, class_thresholds)

            log_jacdet, non_pos_jacdet = compute_jacdet(flow.detach().cpu().float().numpy())
            target_fg = (tgt_oh.sum(dim=1, keepdim=True) > 0).float()
            ndv, per_ndv = compute_ndv(flow, target_fg)

            subject_metrics = {
                'subject': data['subject'][0],
                'log_jacdet': log_jacdet.mean(),
                'np_jacdet': non_pos_jacdet_percent(non_pos_jacdet, flow),
                'ndv': ndv,
                'per_ndv': per_ndv,
            }
            for organ, init_d, reg_d, init_h, reg_h, init_n, reg_n, valid in zip(
                    organ_names, init_dice, reg_dice, init_hd95, reg_hd95,
                    init_nsd, reg_nsd, pair_valid):
                subject_metrics.update({
                    f'valid_{organ}': bool(valid),
                    f'init_dice_{organ}': init_d,
                    f'reg_dice_{organ}': reg_d,
                    f'init_hd95_{organ}': init_h,
                    f'reg_hd95_{organ}': reg_h,
                    f'init_nsd_{organ}': init_n,
                    f'reg_nsd_{organ}': reg_n,
                })

            metrics_list.append(subject_metrics)
            print(subject_metrics)

    df = pd.DataFrame(metrics_list)

    os.makedirs(cfg.save_dir, exist_ok=True)
    # non-default pyramid levels get their own file, so a sweep over --flow-idx
    # does not overwrite the headline numbers
    suffix = '' if cfg.flow_idx == -1 and cfg.scale is None \
        else f'_s{cfg.scale}_f{cfg.flow_idx}'
    df.to_csv(os.path.join(cfg.save_dir, f'metrics_{cfg.epoch_id:03d}{suffix}.csv'),
              index=False)

    print("\nOverall Statistics:")
    print(f"log_jacdet: {df['log_jacdet'].mean():.4f} +/- {df['log_jacdet'].std():.4f}")
    print(f"np_jacdet: {df['np_jacdet'].mean():.4f} +/- {df['np_jacdet'].std():.4f}")
    print(f"ndv: {df['ndv'].mean():.4f} +/- {df['ndv'].std():.4f}")
    print(f"per_ndv: {df['per_ndv'].mean():.4f} +/- {df['per_ndv'].std():.4f}")
    for organ in organ_names:
        valid_count = int(df[f'valid_{organ}'].sum())
        print(f"\n{organ} (valid n={valid_count}/{len(df)}):")
        print(f"Initial Dice: {df[f'init_dice_{organ}'].mean():.4f} +/- {df[f'init_dice_{organ}'].std():.4f}")
        print(f"Registered Dice: {df[f'reg_dice_{organ}'].mean():.4f} +/- {df[f'reg_dice_{organ}'].std():.4f}")
        print(f"Initial HD95: {df[f'init_hd95_{organ}'].mean():.4f} +/- {df[f'init_hd95_{organ}'].std():.4f}")
        print(f"Registered HD95: {df[f'reg_hd95_{organ}'].mean():.4f} +/- {df[f'reg_hd95_{organ}'].std():.4f}")
        print(f"Initial NSD: {df[f'init_nsd_{organ}'].mean():.4f} +/- {df[f'init_nsd_{organ}'].std():.4f}")
        print(f"Registered NSD: {df[f'reg_nsd_{organ}'].mean():.4f} +/- {df[f'reg_nsd_{organ}'].std():.4f}")


if __name__ == '__main__':
    import pathlib
    import configargparse
    from utils import set_seed

    script_dir = os.path.dirname(os.path.abspath(__file__))

    p = configargparse.ArgParser()
    p.add_argument('--method-folder',
                   '-m',
                   required=True,
                   type=lambda f: pathlib.Path(f).absolute(),
                   help='path of method folder')
    p.add_argument('--exp-id',
                   '-exp',
                   required=True,
                   type=int)
    p.add_argument('--epoch-id',
                   '-epoch',
                   required=True,
                   type=int)
    p.add_argument('--split',
                   default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--flow-idx',
                   '-i',
                   default=-1,
                   type=int,
                   help='pyramid level to score: -1 = final flow (default, and '
                        'what the paper reports), 0 = coarsest. Ignored for '
                        'non-pyramid models.')
    p.add_argument('--scale',
                   '-s',
                   default=None,
                   type=int,
                   help='override the registration-head upsampling factor. '
                        'Needed when --flow-idx selects a coarser level, whose '
                        'flow is at 1/scale resolution (e.g. -s 16 -i 0). '
                        'Default: take it from the training config.')
    args = p.parse_args()

    eval_dir = 'eval' if args.split == 'test' else f'eval_{args.split}'
    save_dir = os.path.join(args.method_folder, f'exp{args.exp_id}/{eval_dir}')
    train_cfg = Config.fromfile(os.path.join(args.method_folder, f'exp{args.exp_id}/train_configs.py'))
    load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/{args.epoch_id:04d}.pth')

    config = Config.fromfile(os.path.join(script_dir, 'eval_cfg.py'))
    for key in ['data_dir', 'split_json', 'label_list', 'image_size',
                'cache_rate', 'num_workers', 'batch_size']:
        if key in train_cfg:
            config[key] = train_cfg[key]
    config.update(dict(
        epoch_id=args.epoch_id,
        eval_split=args.split,
        save_dir=save_dir,
        load_model=load_model,
        model_cfg=train_cfg.model_cfg,
        registration_cfg=train_cfg.registration_cfg,
        flow_idx=args.flow_idx,
        scale=args.scale,
    ))
    print(f'[eval] split={args.split} flow_idx={config.flow_idx} '
          f'scale={config.scale or "from config"}')
    set_seed(2023)
    eval(config)
