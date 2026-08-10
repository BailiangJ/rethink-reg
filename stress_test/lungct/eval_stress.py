"""Stress-test a lung-CT model under controlled initial misalignment.

Two stress modes (manuscript S5.9):
  --stress translation  shifts the moving image by a known magnitude
  --stress svf          synthesises the moving image from the fixed one through
                        a random stationary velocity field of known magnitude,
                        which also yields a dense ground-truth flow, hence EPE

The architecture is read back from <method_folder>/exp<N>/train_configs.py, and
the evaluation data from the ordinary evaluator's config under tasks/lungct, so
the stress test scores exactly the same held-out cases.

Lung250M-4B ships landmarks but no lung masks, so the mask-based Dice/HD95, the
lung overlap and the lung-region non-positive-Jacobian rates are NLST-only; the
image foreground stands in for the mask wherever the SVF or the endpoint error
needs one.

Examples:
    python eval_stress.py -m ./pwc_outputs -exp 0 -epoch 100 --stress translation
    python eval_stress.py -m ./pwc_outputs -exp 0 -epoch 100 --stress svf \
                          --dataset lung250m --magnitudes 0,5,10
"""

import os
import pathlib
import sys
from functools import partial
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import configargparse
import numpy as np
import torch
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import DiceMetric, PSNRMetric, SSIMMetric, compute_hausdorff_distance

from models import Warp, build_flow_estimator, build_metrics, build_registration_head
from stress_test.stress_utils import (compute_epe_mm, foreground_overlap_fraction,
                                      generate_svf_flows, infer_raw_flows,
                                      make_translation_vectors, parse_float_list,
                                      synthesize_from_svf,
                                      translate_image_and_keypoints)
from utils import load_data_Lung250M, load_data_NLST, set_seed, worker_init_fn

# dataset key -> (loader, eval config, ships lung masks)
DATASETS = {
    'nlst': (load_data_NLST, 'tasks/lungct/eval_configs/nlst.py', True),
    'lung250m': (partial(load_data_Lung250M, val=False),
                 'tasks/lungct/eval_configs/lung250m.py', False),
}


def _scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0 or np.isnan(value).all():
        return float('nan')
    return float(np.nanmean(value))


def _percentile(value, percentile):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.nanpercentile(value, percentile))


def _append(record, key, value):
    record.setdefault(key, []).append(value)


def _foreground(image):
    """Binary image foreground: the stand-in mask when no segmentation ships."""
    return torch.where(image > 0.0, 1.0, 0.0).float()


def _jacobian_stats(flow, fixed_image, fixed_mask, compute_jacdet):
    """SDlogJ, plus the non-positive-Jacobian rate over foreground and lung.

    fixed_mask may be None, in which case the lung-region rate is not defined.
    """
    jacdet = compute_jacdet(flow.detach().cpu().float().numpy())
    log_jacdet = np.std(np.log((jacdet + 3).clip(1e-9, 1e9)), axis=(1, 2, 3))

    fixed_fg = torch.where(fixed_image > 0.0, 1.0, 0.0).cpu().numpy().squeeze(1)[:, 2:-2, 2:-2, 2:-2]
    non_pos = jacdet <= 0
    fg_np_jacdet = (np.sum(non_pos * fixed_fg) / max(np.sum(fixed_fg), 1)) * 100

    lung_np_jacdet = float('nan')
    if fixed_mask is not None:
        fixed_lung = fixed_mask.cpu().numpy().squeeze(1)[:, 2:-2, 2:-2, 2:-2]
        lung_np_jacdet = (np.sum(non_pos * fixed_lung) / max(np.sum(fixed_lung), 1)) * 100
    return float(np.mean(log_jacdet)), float(fg_np_jacdet), float(lung_np_jacdet)


def _load_model_and_config(args):
    save_dir = os.path.join(args.method_folder, f'exp{args.exp_id}/eval/stress')
    train_cfg = Config.fromfile(os.path.join(args.method_folder, f'exp{args.exp_id}/train_configs.py'))
    if train_cfg.get('model_cfg', False):
        model_cfg = train_cfg.model_cfg
        load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/{args.epoch_id:04d}.pth')
    else:
        model_cfg = train_cfg.reg_model_cfg
        load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/reg_{args.epoch_id:04d}.pth')

    if model_cfg.get('type') == 'TransMorph':
        model_cfg.type = 'TransMorph3D'
        model_cfg.config.spatial_dims = 3

    _, cfg_path, has_masks = DATASETS[args.dataset]
    cfg = Config.fromfile(os.path.join(repo_root, cfg_path))
    cfg.update(dict(
        epoch_id=args.epoch_id,
        save_dir=save_dir,
        load_model=load_model,
        model_cfg=model_cfg,
        registration_cfg=train_cfg.registration_cfg,
        has_masks=has_masks,
    ))
    if args.max_cases is not None:
        cfg.data_indexs = cfg.data_indexs[:args.max_cases]
    return cfg


def infer(cfg, args):
    print(cfg.load_model)
    print(f'[stress] dataset={args.dataset} mode={args.stress}')
    model = build_flow_estimator(cfg.model_cfg)
    model.load_state_dict(torch.load(cfg.load_model, map_location=torch.device(cfg.device)))
    model.to(cfg.device)
    model.eval()

    reg_head = build_registration_head(cfg.registration_cfg).to(cfg.device)
    reg_head.eval()
    warp_linear = Warp(cfg.image_size, interp_mode='bilinear').to(cfg.device)
    warp_nearest = Warp(cfg.image_size, interp_mode='nearest').to(cfg.device)

    load_data = DATASETS[args.dataset][0]
    dataset = load_data(cfg, cache_rate=cfg.cache_rate, num_workers=cfg.num_workers)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, worker_init_fn=worker_init_fn)

    compute_dice = DiceMetric(include_background=False, reduction='mean')
    compute_hd95 = partial(compute_hausdorff_distance, include_background=False, percentile=95)
    compute_ssim = SSIMMetric(data_range=torch.tensor(1.0), spatial_dims=3)._compute_metric
    compute_psnr = PSNRMetric(max_val=1.0)._compute_metric
    compute_jacdet = build_metrics(dict(type='jacdet'))
    compute_tre = build_metrics(cfg.tre_cfg)
    spacing = torch.tensor(cfg.tre_cfg.spacing, dtype=torch.float32, device=cfg.device).view(1, 1, 3)

    levels = parse_float_list(args.magnitudes)
    if not levels:
        raise ValueError('--magnitudes must contain at least one value')

    if args.stress == 'translation':
        translation_vectors = make_translation_vectors(
            levels, cfg.tre_cfg.spacing, len(cfg.data_indexs), seed=args.seed)
    else:
        translation_vectors = None

    records = {}

    with torch.no_grad():
        for case_idx, data in enumerate(dataloader):
            source_orig = data['inp'].float().to(cfg.device)
            target = data['exp'].float().to(cfg.device)
            src_keypnt_orig = data['inp_keypnt'].float().to(cfg.device)
            tgt_keypnt = data['exp_keypnt'].float().to(cfg.device)
            if cfg.has_masks:
                src_mask_orig = data['inp_mask'].float().to(cfg.device)
                tgt_mask = data['exp_mask'].float().to(cfg.device)
            else:
                src_mask_orig = tgt_mask = None
            tgt_fg = tgt_mask if cfg.has_masks else _foreground(target)

            for level_idx, level in enumerate(levels):
                gt_fwd_flow = None
                gt_bck_flow = None
                svf_stats = None
                translation_vox = torch.zeros(3, dtype=torch.float32)
                stress_seed = args.seed

                if args.stress == 'translation':
                    translation_vox = translation_vectors[level_idx, case_idx]
                    source, src_keypnt, src_mask, _ = translate_image_and_keypoints(
                        source_orig, src_keypnt_orig, translation_vox.to(cfg.device),
                        warp_linear, mask=src_mask_orig, mask_warp=warp_nearest)
                elif args.stress == 'svf':
                    stress_seed = args.seed + case_idx * 1009 + level_idx
                    gt_fwd_flow, gt_bck_flow, svf_stats = generate_svf_flows(
                        cfg.image_size,
                        cfg.tre_cfg.spacing,
                        target_mean_mm=level,
                        device=cfg.device,
                        seed=stress_seed,
                        coarse_size=args.svf_coarse_size,
                        smooth_sigma=args.svf_smooth_sigma,
                        int_steps=args.svf_int_steps,
                        calibration_iters=args.svf_calibration_iters,
                        calibration_stat=args.svf_calibration_stat,
                        fg_mask=tgt_fg,
                        fg_smooth_sigma=args.svf_fg_smooth_sigma)
                    source, src_keypnt, src_mask = synthesize_from_svf(
                        target, tgt_keypnt, gt_fwd_flow, gt_bck_flow, warp_linear,
                        fixed_mask=tgt_mask, mask_warp=warp_nearest)
                else:
                    raise KeyError(f'Unsupported stress mode: {args.stress}')

                src_fg = src_mask if cfg.has_masks else _foreground(source)

                init_tre = torch.linalg.norm((tgt_keypnt - src_keypnt) * spacing, dim=-1)
                init_ssim = compute_ssim(source, target)
                init_psnr = compute_psnr(source, target)
                fg_overlap = foreground_overlap_fraction(target, source).detach().cpu().numpy()

                fwd_raw, bck_raw = infer_raw_flows(model, source, target)
                fwd_flow, y_source, _ = reg_head(fwd_raw, source)
                bck_flow, y_target, _ = reg_head(bck_raw, target)

                fwd_tre = compute_tre(fwd_flow, tgt_keypnt, src_keypnt)
                bck_tre = compute_tre(bck_flow, src_keypnt, tgt_keypnt)
                fwd_ssim = compute_ssim(y_source, target)
                bck_ssim = compute_ssim(y_target, source)
                fwd_psnr = compute_psnr(y_source, target)
                bck_psnr = compute_psnr(y_target, source)
                fwd_log_jacdet, fwd_np_jacdet, fwd_lung_np_jacdet = _jacobian_stats(
                    fwd_flow, target, tgt_mask, compute_jacdet)
                bck_log_jacdet, bck_np_jacdet, bck_lung_np_jacdet = _jacobian_stats(
                    bck_flow, source, src_mask, compute_jacdet)

                fwd_epe_mean = np.nan
                fwd_epe_p95 = np.nan
                bck_epe_mean = np.nan
                bck_epe_p95 = np.nan
                gt_disp_mean = level if args.stress == 'translation' else np.nan
                gt_disp_p95 = level if args.stress == 'translation' else np.nan
                gt_disp_max = level if args.stress == 'translation' else np.nan
                if gt_fwd_flow is not None:
                    fwd_epe_mean, fwd_epe_p95 = compute_epe_mm(fwd_flow, gt_fwd_flow, cfg.tre_cfg.spacing, tgt_fg)
                    bck_epe_mean, bck_epe_p95 = compute_epe_mm(bck_flow, gt_bck_flow, cfg.tre_cfg.spacing, src_fg)
                    gt_disp_mean = svf_stats['mean'][0]
                    gt_disp_p95 = svf_stats['p95'][0]
                    gt_disp_max = svf_stats['max'][0]

                _append(records, 'case_idx', case_idx)
                _append(records, 'data_index', cfg.data_indexs[case_idx])
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
                _append(records, 'init_tre', _scalar(init_tre))
                _append(records, 'init_tre_p95', _percentile(init_tre, 95))
                _append(records, 'fwd_tre', _scalar(fwd_tre))
                _append(records, 'fwd_tre_p95', _percentile(fwd_tre, 95))
                _append(records, 'bck_tre', _scalar(bck_tre))
                _append(records, 'bck_tre_p95', _percentile(bck_tre, 95))
                _append(records, 'init_ssim', _scalar(init_ssim))
                _append(records, 'fwd_ssim', _scalar(fwd_ssim))
                _append(records, 'bck_ssim', _scalar(bck_ssim))
                _append(records, 'init_psnr', _scalar(init_psnr))
                _append(records, 'fwd_psnr', _scalar(fwd_psnr))
                _append(records, 'bck_psnr', _scalar(bck_psnr))
                _append(records, 'fwd_log_jacdet', fwd_log_jacdet)
                _append(records, 'bck_log_jacdet', bck_log_jacdet)
                _append(records, 'fwd_np_jacdet', fwd_np_jacdet)
                _append(records, 'bck_np_jacdet', bck_np_jacdet)
                _append(records, 'fwd_epe_mean', _scalar(fwd_epe_mean))
                _append(records, 'fwd_epe_p95', _scalar(fwd_epe_p95))
                _append(records, 'bck_epe_mean', _scalar(bck_epe_mean))
                _append(records, 'bck_epe_p95', _scalar(bck_epe_p95))

                fwd_dice = None
                if cfg.has_masks:
                    y_src_mask = warp_nearest(src_mask, fwd_flow)
                    y_tgt_mask = warp_nearest(tgt_mask, bck_flow)
                    init_dice = compute_dice(src_mask, tgt_mask)
                    init_hd95 = compute_hd95(src_mask, tgt_mask)
                    fwd_dice = compute_dice(y_src_mask, tgt_mask)
                    bck_dice = compute_dice(y_tgt_mask, src_mask)
                    fwd_hd95 = compute_hd95(y_src_mask, tgt_mask)
                    bck_hd95 = compute_hd95(y_tgt_mask, src_mask)
                    lung_overlap = foreground_overlap_fraction(
                        tgt_mask, src_mask, threshold=0.5).detach().cpu().numpy()

                    _append(records, 'lung_overlap', _scalar(lung_overlap))
                    _append(records, 'init_dice', _scalar(init_dice))
                    _append(records, 'fwd_dice', _scalar(fwd_dice))
                    _append(records, 'bck_dice', _scalar(bck_dice))
                    _append(records, 'init_hd95', _scalar(init_hd95))
                    _append(records, 'fwd_hd95', _scalar(fwd_hd95))
                    _append(records, 'bck_hd95', _scalar(bck_hd95))
                    _append(records, 'fwd_lung_np_jacdet', fwd_lung_np_jacdet)
                    _append(records, 'bck_lung_np_jacdet', bck_lung_np_jacdet)

                line = (f"case={case_idx} level={level:g} {args.stress} "
                        f"init_tre={_scalar(init_tre):.2f} fwd_tre={_scalar(fwd_tre):.2f}")
                if fwd_dice is not None:
                    line += f" fwd_dice={_scalar(fwd_dice):.3f}"
                print(line)

    os.makedirs(cfg.save_dir, exist_ok=True)
    output_prefix = args.output_prefix or cfg.prefix
    save_path = os.path.join(cfg.save_dir, f'{output_prefix}_{args.stress}_stress_{cfg.epoch_id:03d}.npz')
    np.savez(save_path,
             dataset=np.array(args.dataset),
             stress=np.array(args.stress),
             seed=np.asarray(args.seed, dtype=np.int64),
             seed_formula=np.array('translation: seed; svf: seed + case_idx * 1009 + level_idx'),
             svf_generator=np.array('cubic_bspline_stationary_velocity'),
             svf_calibration_stat=np.array(args.svf_calibration_stat),
             svf_control_points=np.asarray(args.svf_coarse_size, dtype=np.int64),
             svf_fg_smooth_sigma=np.asarray(args.svf_fg_smooth_sigma, dtype=np.float32),
             magnitudes_mm=np.asarray(levels, dtype=np.float32),
             spacing=np.asarray(cfg.tre_cfg.spacing, dtype=np.float32),
             **{key: np.asarray(value) for key, value in records.items()})
    print(f'Saved stress metrics to {save_path}')


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument('--method-folder', '-m', required=True,
                        type=lambda f: pathlib.Path(f).absolute(),
                        help='path of method folder')
    parser.add_argument('--exp-id', '-exp', required=True, type=int)
    parser.add_argument('--epoch-id', '-epoch', required=True, type=int)
    parser.add_argument('--dataset', '-d', choices=sorted(DATASETS), default='nlst',
                        help='which held-out set to stress (default: nlst)')
    parser.add_argument('--stress', choices=['translation', 'svf'], required=True)
    parser.add_argument('--magnitudes', default='0,5,10,15,20,30',
                        help='comma-separated translation magnitudes or SVF target mean displacements in mm')
    parser.add_argument('--max-cases', type=int, default=None,
                        help='optional limit for quick smoke tests')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--output-prefix', default=None)
    parser.add_argument('--svf-coarse-size', '--svf-control-points', dest='svf_coarse_size', type=int, default=5,
                        help='number of cubic B-spline control points per axis')
    parser.add_argument('--svf-smooth-sigma', type=float, default=0.0,
                        help='optional Gaussian smoothing on the sparse control lattice')
    parser.add_argument('--svf-int-steps', type=int, default=5)
    parser.add_argument('--svf-calibration-iters', type=int, default=3)
    parser.add_argument('--svf-calibration-stat', choices=['mean', 'p95', 'max'], default='p95',
                        help='which displacement statistic the SVF magnitude should match')
    parser.add_argument('--svf-fg-smooth-sigma', type=float, default=5.0,
                        help='Gaussian smoothing sigma for foreground mask used to gate SVF velocity')
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = _load_model_and_config(args)
    infer(cfg, args)
