import os
import argparse
import os
import sys

sys.path.append('../')
import time
import numpy as np
import torch
from monai.data import DataLoader
from monai.metrics import DiceHelper, compute_hausdorff_distance
from models import build_registration_head, build_flow_estimator, build_metrics
from mmengine import Config
from functools import partial
from models.metrics import calc_jac_dets, calc_measurements, get_identity_grid
from utils import (worker_init_fn, PairDataset, load_data_oasis, load_data_adni, load_data_ixi, load_data_LPBA,
                   load_data_Mindboggle)


def infer(cfg):
    print(cfg.load_model)
    model = build_flow_estimator(cfg.model_cfg)
    model.load_state_dict(
        torch.load(cfg.load_model, map_location=torch.device(cfg.device)))
    model.to(cfg.device)
    model.eval()

    # build registration head module
    reg_head = build_registration_head(cfg.registration_cfg)
    reg_head.to(cfg.device)
    reg_head.eval()

    # load data
    datasets = [
        load_data_oasis(cfg.oasis_cfg,
                        val=True,
                        cache_rate=cfg.oasis_cfg.cache_rate,
                        num_workers=cfg.oasis_cfg.num_workers,
                        ),
        load_data_adni(cfg.adni_cfg,
                       cache_rate=cfg.adni_cfg.cache_rate,
                       num_workers=cfg.adni_cfg.num_workers,
                       ),
        load_data_ixi(cfg.ixi_cfg,
                      cache_rate=cfg.ixi_cfg.cache_rate,
                      num_workers=cfg.ixi_cfg.num_workers,
                      ),
        load_data_LPBA(cfg.lpba_cfg,
                       cache_rate=cfg.lpba_cfg.cache_rate,
                       num_workers=cfg.lpba_cfg.num_workers,
                       ),
        load_data_Mindboggle(cfg.mindboggle_cfg,
                             cache_rate=cfg.mindboggle_cfg.cache_rate,
                             num_workers=cfg.mindboggle_cfg.num_workers,
                             ),
    ]
    datasets = [PairDataset(dataset, length=cfg.num_pairs) for dataset in datasets]
    for d in datasets: print('num of pairs from each evaluation dataset:', len(d))
    dataloaders = [DataLoader(dataset,
                              batch_size=1,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              ) for dataset in datasets]
    dataset_names = ['OASIS', 'ADNI', 'IXI', 'LPBA', 'Mindboggle']

    # metrics: sdlogjacdet, nonposjacdet, dice, hd95, ndv
    compute_jacdet = build_metrics(dict(type='sdlogjac'))
    compute_dice = DiceHelper(include_background=False, get_not_nans=True)
    compute_hausdorff = partial(compute_hausdorff_distance, include_background=False, percentile=95)
    metric_funcs = dict(dice=compute_dice,
                        jacdet=compute_jacdet,
                        hd=compute_hausdorff)

    # get_identity_grid((3,H,W,D)) -> voxel-wise grid
    # calc_jac_dets(deformation)
    # calc_measurements(jac_dets, mask(image>0))
    id_grid = get_identity_grid(np.empty((3, *cfg.image_size)))

    os.makedirs(cfg.save_dir, exist_ok=True)
    for ds_name, dataloader in zip(dataset_names, dataloaders):
        print(ds_name)
        per_np_jacdet_list = []
        non_pos_jacdet_list = []
        sd_logjacdet_list = []
        ndv_list = []
        per_ndv_list = []
        reg_dice_list = []
        init_dice_list = []
        hd95_list = []
        runtime_list = []
        torch.cuda.empty_cache()
        for data in dataloader:
            with torch.no_grad():
                source = data[0]['image'].float().to(cfg.device)
                target = data[1]['image'].float().to(cfg.device)
                source_oh = data[0]['label'].float().to(cfg.device)
                target_oh = data[1]['label'].float().to(cfg.device)

                target_fg = torch.where(target > 0.0, 1.0, 0.0)

                start = time.time()
                if cfg.pyramid:
                    fwd_flows, _ = model(source, target)
                    half_res_flow = fwd_flows[-1]
                else:
                    half_res_flow = model(source, target)
                full_res_flow, y_source, y_source_oh = reg_head(half_res_flow, source, source_oh)
                runtime_list.append(time.time() - start)

                # dice
                init_dice, not_nans = metric_funcs['dice'](source_oh, target_oh)
                reg_dice, not_nans = metric_funcs['dice'](y_source_oh, target_oh)
                init_dice_list.append(init_dice.detach().cpu().numpy())
                reg_dice_list.append(reg_dice.detach().cpu().numpy())
                print('init_dsc:', np.mean(init_dice.detach().cpu().numpy()))
                print('reg_dsc:', np.mean(reg_dice.detach().cpu().numpy()))

                # jacdet
                log_jacdet, non_pos_jacdet = compute_jacdet(
                    full_res_flow.detach().cpu().numpy(),
                    target_fg.detach().cpu().numpy())
                non_pos_jacdet_list.append(non_pos_jacdet)
                sd_logjacdet_list.append(log_jacdet)
                per_np_jacdet = non_pos_jacdet / target_fg.sum(dim=(2, 3, 4)).detach().cpu().numpy()
                per_np_jacdet_list.append(per_np_jacdet)
                print('per_np_jacdet:', per_np_jacdet)

                # hd95
                # init_hd = metric_funcs['hd'](source_oh, target_oh)
                # reg_hd = metric_funcs['hd'](y_source_oh, target_oh)
                # hd95_list.append(reg_hd.detach().detach().cpu().numpy())
                # print('hd95:', np.nanmean(reg_hd.detach().cpu().numpy()))
                # NOTE: hausdorff distance metric has memory leakage
                # gc.collect()

                # ndv
                # B=1
                target_fg = target_fg.squeeze().detach().cpu().numpy()[1:-1, 1:-1, 1:-1]
                # full_res_flow: (1,3,H,W,D)
                disp_field = full_res_flow.squeeze().detach().cpu().numpy()
                def_field = disp_field + id_grid
                jac_dets = calc_jac_dets(def_field)
                _, _, ndv, _ = calc_measurements(jac_dets, target_fg)
                total_voxels = np.sum(target_fg)
                per_ndv = ndv / total_voxels
                ndv_list.append(ndv)
                per_ndv_list.append(per_ndv)
                print('ndv:', ndv, 'per_ndv:', per_ndv)

        init_dice_list = np.vstack(init_dice_list)
        reg_dice_list = np.vstack(reg_dice_list)
        # hd95_list = np.vstack(hd95_list)
        non_pos_jacdet_list = np.concatenate(non_pos_jacdet_list, axis=0)
        sd_logjacdet_list = np.concatenate(sd_logjacdet_list, axis=0)
        per_np_jacdet_list = np.concatenate(per_np_jacdet_list, axis=0)
        runtime_list = np.array(runtime_list)
        ndv_list = np.array(ndv_list)
        per_ndv_list = np.array(per_ndv_list)
        print(
            # hd95_list.shape,
            per_np_jacdet_list.shape,
            ndv_list.shape,
            per_ndv_list.shape,
            runtime_list.shape
        )

        np.savez(
            file=os.path.join(cfg.save_dir, ds_name + '.npz'),
            init_dice=init_dice_list,
            reg_dice=reg_dice_list,
            hd95=hd95_list,
            non_pos_jacdet=non_pos_jacdet_list,
            sd_logjacdet=sd_logjacdet_list,
            per_np_jacdet=per_np_jacdet_list,
            ndv=ndv_list,
            per_ndv=per_ndv_list,
            runtime=runtime_list
        )

        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_mb))
        max_mem_re = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_re))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    import pathlib

    import configargparse

    from utils import set_seed

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
    p.add_argument('--pyramid',
                   '-p',
                   required=True,
                   type=str2bool)
    p.add_argument('--random-seed',
                   '-seed',
                   required=True,
                   type=int,
                   help='random seed')

    args = p.parse_args()
    save_dir = os.path.join(args.method_folder, f'exp{args.exp_id}/eval')
    load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/{args.epoch_id:04d}.pth')
    train_cfg = Config.fromfile(os.path.join(args.method_folder, f'exp{args.exp_id}/train_configs.py'))

    # config = Config.fromfile('../configs/eval/_base_.py')
    config = Config.fromfile('./eval_cfg.py')

    config.update(dict(
        save_dir=save_dir,
        load_model=load_model,
        model_cfg=train_cfg.model_cfg,
        registration_cfg=train_cfg.registration_cfg,
        pyramid=args.pyramid,
    ))
    set_seed(args.random_seed)
    infer(config)
