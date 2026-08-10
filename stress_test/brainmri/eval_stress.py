import gc
import json
import os
import pathlib
import sys
import time
from functools import partial
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import configargparse
import nibabel as nib
import numpy as np
import torch
from mmengine import Config
from monai.metrics import DiceHelper, compute_hausdorff_distance
from reorient_nii import reorient

from models.metrics import calc_jac_dets, calc_measurements, get_identity_grid
from models import Warp, build_flow_estimator, build_metrics, build_registration_head
from utils import resolve_path, set_seed

from stress_test.stress_utils import (
    compute_epe_stats_mm, dense_translation_flow, foreground_overlap_fraction,
    generate_svf_flows, infer_raw_flows, make_translation_vectors, parse_float_list,
)

LUMIR_SPACING = (1.0, 1.0, 1.0)
EPE_PERCENTILES = np.arange(5, 101, 5, dtype=np.float32)
DEFAULT_SVF_MAGNITUDES = '5,6,7,8,9,10,11,12,13,14,15'
DEFAULT_TRANSLATION_MAGNITUDES = '0,5,10,15,20,30'

_CLASSES = [
    0,  2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53,
    54, 58, 60,
]


def select_oh(label: torch.Tensor) -> torch.Tensor:
    B, C, H, W, D = label.shape
    oh = torch.zeros((B, len(_CLASSES), H, W, D), device=label.device, dtype=label.dtype)
    sq = label.squeeze(1)
    for i, c in enumerate(_CLASSES):
        oh[:, i][sq == c] = 1
    return oh


def norm_img(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-8)


def _scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0 or np.isnan(value).all():
        return float('nan')
    return float(np.nanmean(value))


def _append(record, key, value):
    record.setdefault(key, []).append(value)


def _append_epe_percentiles(record, prefix, values):
    for percentile, value in zip(EPE_PERCENTILES, values):
        _append(record, f'{prefix}_p{int(percentile):02d}', _scalar(value))


def _load_model_and_config(args):
    save_dir = os.path.join(args.method_folder, f'exp{args.exp_id}/eval/stress')
    train_cfg = Config.fromfile(os.path.join(args.method_folder, f'exp{args.exp_id}/train_configs.py'))
    if train_cfg.get('model_cfg', False):
        model_cfg = train_cfg.model_cfg
        load_model = os.path.join(
            args.method_folder, f'exp{args.exp_id}/saved_models/{args.epoch_id:04d}.pth')
    else:
        model_cfg = train_cfg.reg_model_cfg
        load_model = os.path.join(
            args.method_folder, f'exp{args.exp_id}/saved_models/reg_{args.epoch_id:04d}.pth')

    # resolved against this script's directory, so the command works from
    # anywhere -- the README invokes it from the repository root
    cfg = Config.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'eval_cfg.py'))
    cfg.update(dict(
        epoch_id=args.epoch_id,
        save_dir=save_dir,
        load_model=load_model,
        model_cfg=model_cfg,
        registration_cfg=train_cfg.registration_cfg,
        pyramid=bool(train_cfg.get('scale_pyramid', False)),
    ))
    return cfg


def _load_nii_tensor(path: str, device: str) -> torch.Tensor:
    img = nib.load(path)
    img = reorient(img, orientation='LIA')
    return (torch.from_numpy(np.ascontiguousarray(img.get_fdata()))
            .unsqueeze(0).unsqueeze(0).float().to(device))


def infer(cfg, args):
    data_dir = resolve_path(cfg.data_dir)
    print(cfg.load_model)
    model = build_flow_estimator(cfg.model_cfg)
    model.load_state_dict(torch.load(cfg.load_model, map_location=torch.device(cfg.device)))
    model.to(cfg.device)
    model.eval()

    reg_head = build_registration_head(cfg.registration_cfg).to(cfg.device)
    reg_head.eval()
    warp_linear = Warp(cfg.image_size, interp_mode='bilinear').to(cfg.device)
    warp_nearest = Warp(cfg.image_size, interp_mode='nearest').to(cfg.device)

    compute_dice = DiceHelper(include_background=False, get_not_nans=True)
    compute_hd95 = partial(compute_hausdorff_distance, include_background=False, percentile=95)
    compute_sdlogjac = build_metrics(dict(type='sdlogjac'))
    id_grid = get_identity_grid(np.empty((3, *cfg.image_size)))

    with open(resolve_path(cfg.json_file)) as f:
        pairs_dict = json.load(f)['validation']
    if args.max_cases is not None:
        pairs_dict = pairs_dict[:args.max_cases]

    levels = parse_float_list(args.magnitudes)
    if not levels:
        raise ValueError('--magnitudes must contain at least one value')

    if args.stress == 'translation':
        translation_vectors = make_translation_vectors(
            levels, LUMIR_SPACING, len(pairs_dict), seed=args.seed)
    else:
        translation_vectors = None

    records = {}

    with torch.no_grad():
        for case_idx, d in enumerate(pairs_dict):
            torch.cuda.empty_cache()

            fixed_path = os.path.join(data_dir, d['fixed'])
            moving_path = os.path.join(data_dir, d['moving'])
            fixed_file = os.path.basename(fixed_path)
            moving_file = os.path.basename(moving_path)
            fixed_label_path = os.path.join(data_dir, 'labelsVal', fixed_file)
            moving_label_path = os.path.join(data_dir, 'labelsVal', moving_file)

            target_label_scalar = _load_nii_tensor(fixed_label_path, cfg.device)
            source_label_scalar = _load_nii_tensor(moving_label_path, cfg.device)
            target_oh = select_oh(target_label_scalar)

            target_raw = _load_nii_tensor(fixed_path, cfg.device)
            source_raw = _load_nii_tensor(moving_path, cfg.device)
            target_fg = torch.where(target_raw > 0.0, 1.0, 0.0)
            target = norm_img(target_raw)
            source = norm_img(source_raw)
            del target_raw, source_raw

            fg_count = float(target_fg.sum())
            target_fg_np = target_fg.detach().cpu().numpy()
            target_fg_trim = target_fg_np.squeeze()[1:-1, 1:-1, 1:-1]
            fg_count_trim = float(np.sum(target_fg_trim))

            for level_idx, level in enumerate(levels):
                gt_fwd_flow = None
                gt_bck_flow = None
                svf_stats = None
                translation_vox = torch.zeros(3, dtype=torch.float32)
                stress_seed = args.seed

                if args.stress == 'translation':
                    translation_vox = translation_vectors[level_idx, case_idx]
                    flow_t = dense_translation_flow(cfg.image_size, translation_vox.to(cfg.device), cfg.device)
                    source_stressed = warp_linear(source, flow_t)
                    source_label_stressed = warp_nearest(source_label_scalar, flow_t)
                    source_oh_stressed = select_oh(source_label_stressed)
                    del flow_t, source_label_stressed
                elif args.stress == 'svf':
                    stress_seed = args.seed + case_idx * 1009 + level_idx
                    gt_fwd_flow, gt_bck_flow, svf_stats = generate_svf_flows(
                        cfg.image_size, LUMIR_SPACING,
                        target_mean_mm=level,
                        device=cfg.device,
                        seed=stress_seed,
                        coarse_size=args.svf_coarse_size,
                        smooth_sigma=args.svf_smooth_sigma,
                        int_steps=args.svf_int_steps,
                        calibration_iters=args.svf_calibration_iters,
                        calibration_stat=args.svf_calibration_stat,
                        fg_mask=target_fg,
                        fg_smooth_sigma=args.svf_fg_smooth_sigma)
                    source_stressed = warp_linear(target, gt_bck_flow)
                    source_label_stressed = warp_nearest(target_label_scalar, gt_bck_flow)
                    source_oh_stressed = select_oh(source_label_stressed)
                    del source_label_stressed, gt_bck_flow
                else:
                    raise KeyError(f'Unsupported stress mode: {args.stress}')

                fg_overlap = foreground_overlap_fraction(target, source_stressed).detach().cpu().numpy()
                init_dice, _ = compute_dice(source_oh_stressed, target_oh)
                init_hd95 = compute_hd95(source_oh_stressed, target_oh)

                start_time = time.time()
                fwd_raw, _ = infer_raw_flows(model, source_stressed, target)
                full_res_flow, y_source, y_source_oh = reg_head(fwd_raw, source_stressed, source_oh_stressed)
                elapsed_time = time.time() - start_time
                del fwd_raw, y_source, source_stressed, source_oh_stressed

                fwd_dice, _ = compute_dice(y_source_oh, target_oh)
                fwd_hd95 = compute_hd95(y_source_oh, target_oh)
                del y_source_oh

                flow_np = full_res_flow.detach().cpu().float().numpy()
                log_jacdet_arr, non_pos_arr = compute_sdlogjac(flow_np, target_fg_np)
                log_jacdet_val = float(log_jacdet_arr[0])
                non_pos_val = float(non_pos_arr[0])
                per_np_jacdet_val = non_pos_val / max(fg_count, 1.0)

                def_field = flow_np.squeeze() + id_grid
                jac_dets = calc_jac_dets(def_field)
                _, _, ndv, _ = calc_measurements(jac_dets, target_fg_trim)
                per_ndv = ndv / max(fg_count_trim, 1.0)

                fwd_epe_mean = np.nan
                fwd_epe_percentiles = np.full(len(EPE_PERCENTILES), np.nan, dtype=np.float32)
                fwd_epe_fg_mean = np.nan
                fwd_epe_fg_percentiles = np.full(len(EPE_PERCENTILES), np.nan, dtype=np.float32)
                gt_disp_mean = level if args.stress == 'translation' else np.nan
                gt_disp_p95 = level if args.stress == 'translation' else np.nan
                gt_disp_max = level if args.stress == 'translation' else np.nan
                if gt_fwd_flow is not None:
                    epe_stats = compute_epe_stats_mm(
                        full_res_flow, gt_fwd_flow, LUMIR_SPACING,
                        percentiles=EPE_PERCENTILES)
                    epe_fg_stats = compute_epe_stats_mm(
                        full_res_flow, gt_fwd_flow, LUMIR_SPACING,
                        mask=target_fg, percentiles=EPE_PERCENTILES)
                    fwd_epe_mean = float(epe_stats['mean'][0])
                    fwd_epe_percentiles = epe_stats['percentiles'][0]
                    fwd_epe_fg_mean = float(epe_fg_stats['mean'][0])
                    fwd_epe_fg_percentiles = epe_fg_stats['percentiles'][0]
                    gt_disp_mean = svf_stats['mean'][0]
                    gt_disp_p95 = svf_stats['p95'][0]
                    gt_disp_max = svf_stats['max'][0]
                del full_res_flow

                _append(records, 'case_idx', case_idx)
                _append(records, 'fixed_file', fixed_file)
                _append(records, 'moving_file', moving_file)
                _append(records, 'level_idx', level_idx)
                _append(records, 'level_mm', level)
                _append(records, 'stress_seed', stress_seed)
                _append(records, 'translation_vox_x', float(translation_vox[0]))
                _append(records, 'translation_vox_y', float(translation_vox[1]))
                _append(records, 'translation_vox_z', float(translation_vox[2]))
                _append(records, 'fg_overlap', _scalar(fg_overlap))
                _append(records, 'gt_disp_mean', _scalar(gt_disp_mean))
                _append(records, 'gt_disp_p95', _scalar(gt_disp_p95))
                _append(records, 'gt_disp_max', _scalar(gt_disp_max))
                _append(records, 'init_dice', _scalar(init_dice))
                _append(records, 'fwd_dice', _scalar(fwd_dice))
                _append(records, 'init_hd95', _scalar(init_hd95))
                _append(records, 'fwd_hd95', _scalar(fwd_hd95))
                _append(records, 'fwd_log_jacdet', log_jacdet_val)
                _append(records, 'fwd_np_jacdet', non_pos_val)
                _append(records, 'fwd_per_np_jacdet', per_np_jacdet_val)
                _append(records, 'fwd_ndv', float(ndv))
                _append(records, 'fwd_per_ndv', per_ndv)
                _append(records, 'reg_time_s', elapsed_time)
                _append(records, 'fwd_epe_mean', _scalar(fwd_epe_mean))
                _append_epe_percentiles(records, 'fwd_epe', fwd_epe_percentiles)
                _append(records, 'fwd_epe_fg_mean', _scalar(fwd_epe_fg_mean))
                _append_epe_percentiles(records, 'fwd_epe_fg', fwd_epe_fg_percentiles)

                print(
                    f'case={case_idx} level={level:g} {args.stress} '
                    f'init_dice={_scalar(init_dice):.3f} fwd_dice={_scalar(fwd_dice):.3f}'
                )

            del target_label_scalar, source_label_scalar, target_oh, target, source, target_fg
            gc.collect()

    os.makedirs(cfg.save_dir, exist_ok=True)
    output_prefix = args.output_prefix or 'LUMIR'
    save_path = os.path.join(
        cfg.save_dir, f'{output_prefix}_{args.stress}_stress_{cfg.epoch_id:03d}.npz')
    np.savez(save_path,
             stress=np.array(args.stress),
             seed=np.asarray(args.seed, dtype=np.int64),
             seed_formula=np.array('translation: seed; svf: seed + case_idx * 1009 + level_idx'),
             magnitudes_mm=np.asarray(levels, dtype=np.float32),
             spacing=np.asarray(LUMIR_SPACING, dtype=np.float32),
             epe_percentile_levels=np.asarray(EPE_PERCENTILES, dtype=np.float32),
             epe_fg_mask=np.array('target_intensity_gt_0'),
             svf_calibration_stat=np.array(args.svf_calibration_stat),
             svf_fg_smooth_sigma=np.asarray(args.svf_fg_smooth_sigma, dtype=np.float32),
             **{key: np.asarray(value) for key, value in records.items()})
    print(f'Saved stress metrics to {save_path}')


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument('--method-folder', '-m', required=True,
                        type=lambda f: pathlib.Path(f).absolute(),
                        help='path of method folder')
    parser.add_argument('--exp-id', '-exp', required=True, type=int)
    parser.add_argument('--epoch-id', '-epoch', required=True, type=int)
    parser.add_argument('--stress', choices=['translation', 'svf'], required=True)
    parser.add_argument('--magnitudes', default=None,
                        help='comma-separated magnitudes in mm; defaults are '
                             f'svf={DEFAULT_SVF_MAGNITUDES}, '
                             f'translation={DEFAULT_TRANSLATION_MAGNITUDES}')
    parser.add_argument('--max-cases', type=int, default=None,
                        help='optional limit for quick smoke tests')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--output-prefix', default=None)
    parser.add_argument('--svf-coarse-size', type=int, default=16)
    parser.add_argument('--svf-smooth-sigma', type=float, default=1.0)
    parser.add_argument('--svf-int-steps', type=int, default=5)
    parser.add_argument('--svf-calibration-iters', type=int, default=3)
    parser.add_argument('--svf-calibration-stat', choices=['mean', 'p95', 'max'], default='p95',
                        help='which displacement statistic the SVF magnitude should match')
    parser.add_argument('--svf-fg-smooth-sigma', type=float, default=5.0,
                        help='Gaussian sigma (voxels) for smoothing the foreground mask before '
                             'weighting the SVF velocity field; 0 disables masking')
    args = parser.parse_args()
    if args.magnitudes is None:
        if args.stress == 'svf':
            args.magnitudes = DEFAULT_SVF_MAGNITUDES
        elif args.stress == 'translation':
            args.magnitudes = DEFAULT_TRANSLATION_MAGNITUDES

    set_seed(args.seed)
    cfg = _load_model_and_config(args)
    infer(cfg, args)
