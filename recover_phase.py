import argparse
import glob
import os
import pickle
import re

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import heapq

from skimage.restoration import unwrap_phase
import wandb
import pandas as pd
from collections import defaultdict

from noise_models import PGNoiseModel, generate_noise_sweep
from recovery_algorithms import integrate_coprime_bfs_v3
import constants
from utils import (
    are_coprime,
    compute_phase_diffs_per_shift,
    load_amplitude_no_shift,
    load_shifted_measurements,
    max_in_center_window,
    phasor_phase_error,
    unrap_tesnor,
)


DEFAULT_CROP_SIZE = 1024
RUN= os.getenv('RUN_ID', '0')
DO_AVG = os.getenv('DO_AVG', '1') == '1'
AVG_MOD = int(os.getenv('AVG_MOD', '0'))

results_dir = "results"
DEBUG = False


@torch.no_grad()
def best_vertical_index_only_center_col(diff_v, gt_phasor, s, j0=None):
    """
    Compare indexing only:
      A) use dv at source: dv[i,j]
      B) use dv at destination: dv[i+s,j]
    Assumes dv already points in the correct direction (no conjugate/sign flips).
    """
    H, W = gt_phasor.shape
    if j0 is None:
        j0 = W // 2

    gt_col = gt_phasor[:, j0]
    dv_col = diff_v[:, j0]

    # GT edge phasor for DOWN: (i -> i+s)
    gt_edge = gt_col[s:] * torch.conj(gt_col[:-s])  # length H-s

    # index-only candidates
    pred_src  = dv_col[:-s]   # dv[i]
    pred_dest = dv_col[s:]    # dv[i+s]

    err_src  = torch.angle(pred_src  * torch.conj(gt_edge)).abs().mean().item()
    err_dest = torch.angle(pred_dest * torch.conj(gt_edge)).abs().mean().item()

    print(f"s={s}, j0={j0}:  dv[i] err={err_src:.6f}   dv[i+s] err={err_dest:.6f}")
    return ("src" if err_src < err_dest else "dest"), {"err_src": err_src, "err_dest": err_dest, "j0": j0, "s": s}


def get_gt_phase(exp_name, results_folder):
    gt_path = os.path.join(
        results_folder, f"{exp_name}_ground_truth_amp_phase.npz"
    )
    
    data = np.load(gt_path)
    gt_phase = data["phase"]  # shape (H, W)
    gt_phase = center_crop(gt_phase)

    return gt_phase


def compute_phase_recovery_error(phi_est, exp_name, results_folder, phi_est_ls):
    gt_phase = get_gt_phase(exp_name, results_folder)

    # Convert to numpy for backward compatible metrics
    if isinstance(phi_est, torch.Tensor):
        phi_est_np = phi_est.detach().cpu().numpy()
    else:
        phi_est_np = phi_est

    if phi_est_np.shape != gt_phase.shape:
        raise ValueError(f"Shape mismatch: est {phi_est_np.shape}, gt {gt_phase.shape}")

    if isinstance(gt_phase, np.ndarray):
        gt_phase_t = torch.from_numpy(gt_phase).float()
    else:
        gt_phase_t = gt_phase.float()

    phi_est_t = torch.from_numpy(phi_est_np).float()

    phasor_est = torch.exp(1j * phi_est_t.to(torch.complex64))
    phasor_est_ls = torch.exp(1j * phi_est_ls.cpu().to(torch.complex64))
    phasor_gt = torch.exp(1j * gt_phase_t.to(torch.complex64))

    phase_error_aligned = phasor_phase_error(phasor_est, phasor_gt)
    phase_error_aligned_ls = phasor_phase_error(phasor_est_ls, phasor_gt)

    # Phase is only recoverable up to a global additive constant, so align the recovered
    # phasor to the ground truth's global phase reference (best-fit constant offset, via
    # the circular mean) before computing error -- otherwise mae/mse/error_map are
    # dominated by that arbitrary offset instead of the actual recovery error.
    phasor_diff = phasor_est * torch.conj(phasor_gt)
    global_offset = torch.angle(torch.mean(phasor_diff))
    angular_error = torch.angle(phasor_diff * torch.exp(-1j * global_offset))
    angular_error_np = angular_error.numpy()

    mae = float(np.mean(np.abs(angular_error_np)))
    mae_std = float(np.std(angular_error_np))
    mse = float(np.sqrt(np.mean(angular_error_np**2)))
    circ_mae = float(np.mean(1 - np.cos(angular_error_np)))

    # Recovered phase re-referenced to the ground truth's global phase, for display.
    phi_aligned = ((phi_est_t - global_offset + np.pi) % (2 * np.pi) - np.pi).numpy()

    return {
        "mae": mae,
        "mae_std": mae_std,
        "mse": mse,
        "circ_mae": circ_mae,

        "error_map": np.abs(angular_error_np),
        "phasor_error": float(phase_error_aligned),
        "phasor_error_ls": float(phase_error_aligned_ls),
        "global_offset": float(global_offset),
        "phi_aligned": phi_aligned,
    }, gt_phase


def get_error_per_hop(hops, path_counts, error_map):
    
    hop_to_sum = defaultdict(float)
    hop_count_vals  = defaultdict(int)
    hop_to_count = defaultdict(int)

    # Flatten for efficient iteration
    hops_flat = hops.flatten()
    errors_flat = error_map.flatten()
    counts_flat = path_counts.flatten()

    for h, e, c in zip(hops_flat, errors_flat, counts_flat):
        hop_to_sum[int(h)] += float(e)
        hop_to_count[int(h)] += 1
        hop_count_vals[int(h)] += int(c)

    # Build final structure
    result = {}
    for h in sorted(hop_to_sum.keys()):
        total = hop_to_sum[h]
        path_counts = hop_count_vals[h]
        count = hop_to_count[h]
        result[h] = {
            "sum_error": total,
            "count": count,
            "avg_error": total / count if count > 0 else np.nan,
            "path_counts": path_counts
        }

    return result

_UNSET = object()

def center_crop(array, crop_size = _UNSET):
    if crop_size is _UNSET:
        crop_size = DEFAULT_CROP_SIZE
    if crop_size is None:
        return array

    H, W = array.shape[-2], array.shape[-1]
    c_h, c_w = H // 2, W // 2

    # c_h -= 100
    # c_w -= 50
    
    c_h_min, c_h_max = c_h-crop_size, c_h+crop_size
    c_w_min, c_w_max =c_w-crop_size, c_w+crop_size

    # setup min max bounds 
    c_h_min, c_h_max = max(0, c_h_min), min(H, c_h_max)
    c_w_min, c_w_max  = max(0, c_w_min), min(W, c_w_max)

    return array[..., c_h_min:c_h_max, c_w_min: c_w_max]


def parse_shift_groups(spec):
    """Parse 'shifts;shifts;...' like '1;16,17;23,31,16,17' into a list of int lists."""
    return [[int(s) for s in group.split(",")] for group in spec.split(";")]


def parse_args():
    parser = argparse.ArgumentParser(description="Recover phase from simulated coprime-shift PSI measurements.")
    parser.add_argument("--meas-root", default="./measurements/simulated/",
                         help="Root folder of simulated measurements (matches gen_simulated_data.py --output-dir).")
    parser.add_argument("--results-dir", default="results",
                         help="Where to write recovered-phase figures/analysis.")
    parser.add_argument("--phase-types", default="quad,random,peak",
                         help="Comma-separated subset of: quad,random,peak")
    parser.add_argument("--shift-groups", default=None,
                         help="Semicolon-separated shift groups, e.g. '1;16,17'. Defaults to constants.SHIFT_GROUPS.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False,
                         help="Save extra per-shift diagnostic figures (phase diffs, hop plots, etc). "
                              "Off by default; only the main output (summary figure, recovered fields, "
                              "error CSV) is generated.")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE,
                         help="Center-crop size applied to loaded measurements/ground truth.")
    parser.add_argument("--offset-h", default="0", help="Comma-separated horizontal carrier offsets to sweep.")
    parser.add_argument("--offset-v", default="0", help="Comma-separated vertical carrier offsets to sweep.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MEAS_ROOT = args.meas_root
    results_dir = args.results_dir
    DEBUG = args.debug
    DEFAULT_CROP_SIZE = args.crop_size

    img_folders = sorted([d for d in os.listdir(MEAS_ROOT) if os.path.isdir(os.path.join(MEAS_ROOT, d))])

    print("Found", len(img_folders), f"image folders in {MEAS_ROOT}")

    for img_idx, img_folder in enumerate(img_folders):
        meas_path = os.path.join(MEAS_ROOT, img_folder)
        exp_specs = args.phase_types.split(",")


        for tag in exp_specs:
            base_exp_name = f"Exp_SIM_noisy_{tag}_phase_map"

            # Discover whatever noise levels were actually generated on disk, rather than
            # assuming a fixed sweep count (avoids "File not found" when generation used a
            # different --noise-sweep-n, or --no-add-noise, e.g. a quick noiseless run).
            noise_folders = sorted(
                glob.glob(os.path.join(meas_path, f"{base_exp_name}_noise_std=*_tau=*"))
                + glob.glob(os.path.join(meas_path, f"{base_exp_name}_noiseless"))
            )
            if not noise_folders:
                print(f"No measurement folders found for {base_exp_name} under {meas_path}, skipping.")
                continue

            shift_lists = parse_shift_groups(args.shift_groups) if args.shift_groups else constants.SHIFT_GROUPS

            offset_Hs = [float(x) for x in args.offset_h.split(",")]
            offset_Vs = [float(x) for x in args.offset_v.split(",")]

            USE_WANDB = os.getenv('USE_WANDB', '0') == '1'
            if USE_WANDB:
                wandb_config = {
                    "base_exp_name": base_exp_name,
                    "meas_path": meas_path,
                    "all_shifts": shift_lists
                }
                wandb.init(
                    project="wavefront-sensing",
                    name= "EXP_DEC23_" + base_exp_name + f"with_avg_{DEFAULT_CROP_SIZE}_img" + f"_run{RUN}",
                    config = wandb_config
                )


            for noise_folder in noise_folders:
                data_path = os.path.basename(noise_folder)
                noise_str = data_path[len(base_exp_name) + 1:]  # strip "{base_exp_name}_" prefix
                m = re.search(r"noise_std=([\d.]+)_tau=([\d.]+)", noise_str)
                if m:
                    noise_param = {"noise_std": float(m.group(1)), "tau": float(m.group(2))}
                else:  # "_noiseless" folder
                    noise_param = {"noise_std": 0.0, "tau": float("inf")}
                exp_name = f"{img_folder}_{base_exp_name}_{noise_str}"

                error_per_hop_all_shift = {}
                errors = {
                    "shift_pairs": [],
                    "noise_params": [],
                    "mae": [],
                    "mae_std": [],
                    "mse": [],
                    "circ_mae": [],
                    "phasor_error_ls": [],
                    "phasor_error": [],
                    "avg_hops": []
                }

                if USE_WANDB:
                    phase_errors_table = wandb.Table(columns=["step", "shift", "mae", "mse", "circ_mae", "log_key"])

                for shift_list in shift_lists:

                    if not are_coprime(shift_list): continue 

                    shifts_str = '_'.join([str(s) for s in shift_list])

                    os.makedirs(f'{results_dir}/{exp_name}', exist_ok=True)
                    
                    # print(f"Loading measurements for shift: {shift_list}")
                    measurements = torch.from_numpy(
                        load_shifted_measurements(f"{meas_path}/{data_path}", shift_list) #  / /
                    ).to(device) 
                    
                    c_h, c_w = measurements.shape[-2]//2, measurements.shape[-1]//2
                    measurements = center_crop(measurements)
                    
                    H, W = measurements.shape[-2], measurements.shape[-1]
                    
                    start_i, start_j =  max_in_center_window(measurements[0,0,0].cpu(), 512, 512)
                    
                    for OFFSET_H in offset_Hs:
                        for OFFSET_V in offset_Vs:
                            # print(f"Using offsets H: {OFFSET_H}, V: {OFFSET_V}")

                            grad_h, grad_v, amp_h, amp_v = compute_phase_diffs_per_shift(measurements, shift_list, offset_H=OFFSET_H, offset_V=OFFSET_V, device=device)
                            
                            try:
                                gt_phase = torch.from_numpy(get_gt_phase(base_exp_name, meas_path))
                                gt_phasor = torch.exp(1j * gt_phase)
                            except Exception as e:
                                # print("GT phase loading failed:", e)
                                gt_phasor = None

                            if DEBUG:
                                for i in range(len(grad_h)):
                                    plt.figure(figsize=(12, 4))
                                    plt.subplot(131)
                                    plt.imshow(torch.angle(grad_h[i]).cpu(), cmap='twilight')
                                    plt.title(f'Horizontal phase diff (shift={shift_list[i]})')
                                    plt.colorbar()

                                    plt.subplot(132)
                                    plt.imshow(torch.angle(grad_v[i]).cpu(), cmap='twilight')
                                    plt.title(f'Vertical phase diff (shift={shift_list[i]})')
                                    plt.colorbar()
                                    plt.savefig(f'{results_dir}/{exp_name}/phase_diff_shift{shift_list[i]}_{shifts_str}.png', dpi=150)
                                    plt.close()

                            shifts_array = np.array(shift_list)
                            phases_h = []
                            phases_v = []

                            # plot grad_h for start_i row 
                            # plot side by side shift 0 and 1 
                            def circular_mean(z):
                                return torch.angle(torch.mean(z / torch.abs(z)))
                            if DEBUG:
                                phi_H, phi_V = [], []
                                for i, s in enumerate(shift_list):
                                    plt.figure(figsize=(12, 4))
                                    plt.subplot(121)
                                    plt.plot(torch.angle(grad_h[i][start_i,:-s].cpu()), label=f'shift={s}')

                                    sh = shift_list[i]
                                    if gt_phasor is not None:
                                        gt_diff_h = torch.angle(gt_phasor[:, :-sh] * gt_phasor[:, sh:].conj() ) 
                                        # gt_diff_v = torch.angle(gt_phasor[sh:, :] * gt_phasor[:-sh, :].conj() ) 
                                        gt_diff_v = torch.angle(gt_phasor[:-sh, :].conj()  * gt_phasor[sh:, :])

                                    
                                        plt.plot(gt_diff_h[start_i].cpu(), label=f'GT H shift={s}')
                                    plt.legend()
                                    plt.subplot(122)
                                    plt.plot(torch.angle(grad_v[i][s:,start_j].cpu()), label=f'shift={s}')
                                    
                                    if gt_phasor is not None:
                                        plt.plot(gt_diff_v[:, start_j].cpu(), label=f'GT V shift={s}')
                                
                                    plt.savefig(f'{results_dir}/{exp_name}/phase_diff_profile_shift{s}_{shifts_str}.png', dpi=150)
                                    plt.close()
                                    phase_h_mean = torch.angle(grad_h[i][start_i, 750:1250].mean()).item()
                                    phase_v_mean = torch.angle(grad_v[i][400:800, start_j].mean()).item()
                                    phases_h.append(phase_h_mean)
                                    phases_v.append(phase_v_mean)

                                    phi_H.append(circular_mean(grad_h[i][start_i, :]).item())
                                    phi_V.append(circular_mean(grad_v[i][:, start_j]).item())

                            phi, metadata = integrate_coprime_bfs_v3(grad_h, grad_v, shift_list, amp_h, amp_v, do_avg = DO_AVG,
                                                                    device=device, start_pos = (start_i, start_j), gt_phasor =gt_phasor,
                                                                    results_dir=results_dir, exp_name=exp_name)

                            amplitude = load_amplitude_no_shift(f"{meas_path}/{data_path}")

                            # amp_stack = torch.stack([torch.stack([ah, av]).mean(0) for ah, av in zip(amp_h, amp_v)])
                            # amplitude = torch.median(amp_stack, dim=0).values.cpu().numpy()

                            amplitude = center_crop(amplitude)
                            # exit()
                            # use least squares phase
                            # phi = metadata['phi_least_square']
                            if gt_phasor is not None:
                                phase_error, gt_phase = compute_phase_recovery_error(phi, base_exp_name, meas_path, metadata['phi_least_square'])
                            
                                # get hopsize error 
                                hops, path_counts, error_map = metadata['hops'].cpu(), metadata['path_counts'].cpu(), phase_error['error_map']
                                error_per_hop = get_error_per_hop(hops, path_counts, error_map)

                                # Summary figure: ground truth vs. recovered phase, for this shift set.
                                # Recovered phase is shown re-referenced to the GT's global phase
                                # (phase is only recoverable up to a global additive constant).
                                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                                im0 = axes[0].imshow(gt_phase, cmap='twilight')
                                axes[0].set_title('Ground truth phase')
                                axes[0].axis('off')
                                fig.colorbar(im0, ax=axes[0], fraction=0.046)

                                im1 = axes[1].imshow(phase_error['phi_aligned'], cmap='twilight')
                                axes[1].set_title(f'Recovered phase (shifts={shifts_str})')
                                axes[1].axis('off')
                                fig.colorbar(im1, ax=axes[1], fraction=0.046)

                                fig.suptitle(f"MAE = {phase_error['mae']:.4f} rad, phasor error = {phase_error['phasor_error']:.4f} rad")
                                fig.tight_layout()
                                fig.savefig(f'{results_dir}/{exp_name}/summary_{shifts_str}.png', dpi=200)
                                plt.close(fig)

                                for hop, stats in error_per_hop.items():
                                    if hop not in error_per_hop_all_shift:
                                        error_per_hop_all_shift[hop] = {"sum_error": 0.0, "count": 0, "path_counts": 0}

                                    error_per_hop_all_shift[hop]["sum_error"] += stats["sum_error"]
                                    error_per_hop_all_shift[hop]["path_counts"] += stats["path_counts"]
                                    error_per_hop_all_shift[hop]["count"] += stats["count"]

                                if DEBUG:
                                    save_path = f"{results_dir}/{exp_name}/error_by_hop_{shifts_str}.pkl"
                                    with open(save_path, "wb") as f:
                                        pickle.dump(error_per_hop, f)

                                    # Plot hops
                                    hops_sorted = sorted(error_per_hop.keys())
                                    all_hops_sorted = sorted(error_per_hop_all_shift.keys())
                                    avg_errors = [error_per_hop[h]["avg_error"] for h in hops_sorted]
                                    avg_errors_all_shft = [error_per_hop_all_shift[h]["sum_error"]/error_per_hop_all_shift[h]["count"] for h in all_hops_sorted]
                                    all_counts = [error_per_hop_all_shift[h]["count"] for h in all_hops_sorted]
                                    all_path_counts = [error_per_hop_all_shift[h]["path_counts"]/error_per_hop_all_shift[h]["count"] for h in all_hops_sorted]

                                    plt.figure(figsize=(8, 5))
                                    plt.plot(hops_sorted, avg_errors, marker='o')
                                    plt.xlabel("Hop count")
                                    plt.ylabel("Average error")
                                    plt.title(f"Error vs Hop Count ({shifts_str})")
                                    plt.grid(True)
                                    plot_path = f"{results_dir}/{exp_name}/error_vs_hop_{shifts_str}.png"
                                    plt.savefig(plot_path, dpi=150)
                                    plt.close()

                                    plt.figure(figsize=(8, 5))
                                    plt.plot(all_hops_sorted, avg_errors_all_shft, marker='o')
                                    plt.xlabel("Hop count")
                                    plt.ylabel("Average error")
                                    plt.title(f"Error vs Hop Count ")
                                    plt.grid(True)
                                    plot_path = f"{results_dir}/{exp_name}/error_vs_hop_avg.png"
                                    plt.savefig(plot_path, dpi=150)
                                    plt.close()

                                    plt.figure(figsize=(8, 5))
                                    plt.plot(all_hops_sorted, all_path_counts, marker='o')
                                    plt.xlabel("Hops")
                                    plt.ylabel("Avg #of paths")
                                    plt.title(f"#Paths vs Hop Count ")
                                    plt.grid(True)
                                    plot_path = f"{results_dir}/{exp_name}/path_count_vs_hop_avg.png"
                                    plt.savefig(plot_path, dpi=150)
                                    plt.close()

                                    plt.figure(figsize=(8, 5))
                                    plt.plot(all_hops_sorted, all_counts, marker='o')
                                    plt.xlabel("Hops")
                                    plt.ylabel("Number of pixels")
                                    plt.title(f"#pixels vs Hop Count ")
                                    plt.grid(True)
                                    plot_path = f"{results_dir}/{exp_name}/pixel_count_vs_hop_avg.png"
                                    plt.savefig(plot_path, dpi=150)
                                    plt.close()


                                # Expected phase difference
                                expected_diff_h = gt_phase[:, 1:] - gt_phase[:, :-1]
                                expected_diff_v = gt_phase[1:, :] - gt_phase[:-1, :]

        

                            if DEBUG:
                                # plot sample row and col lines from phi 
                                plt.figure(figsize=(12, 5))
                                plt.subplot(2, 2, 1)
                                plt.plot(phi[start_i, 1000:1250].cpu(), label='Row line at start_i')
                                plt.title('Sample Row Line from Recovered Phase')
                                plt.subplot(2, 2, 2)
                                plt.plot(phi[:, start_j].cpu(), label='Column line at start_j')
                                plt.title('Sample Column Line from Recovered Phase')

                                
                                plt.subplot(2, 2, 3)
                                plt.plot(amplitude[start_i, 1000:1250], label='Row line at start_i')
                                plt.title('Amplitude row')
                                plt.subplot(2, 2, 4)
                                plt.plot(amplitude[:, start_j], label='Column line at start_j')
                                plt.title('Ampltiude col')


                                if gt_phasor is not None:
                                    plt.subplot(2, 2, 3)
                                    plt.plot(gt_phase[start_i], label='GT Row line at start_i')
                                    plt.title('GT Sample Row Line from Recovered Phase')
                                    plt.subplot(2, 2, 4)
                                    plt.plot(gt_phase[:, start_j], label='GT Column line at start_j')
                                    plt.title('GT Sample Column Line from Recovered Phase')
                                    plt.subplot(2, 2, 1)
                                    plt.plot(gt_phase[start_i], label='GT Row line at start_i')
                                    plt.title('GT Sample Row Line from Recovered Phase')
                                    plt.subplot(2, 2, 2)
                                    plt.plot(gt_phase[:, start_j], label='GT Column line at start_j')
                                    plt.title('GT Sample Column Line from Recovered Phase')
                                
                                plt.tight_layout()
                                plt.savefig(f'{results_dir}/{exp_name}/phase_profile_row_col_{shifts_str}.png', dpi=150)
                                plt.close()

                            # print unwrapped min max
                            # does phi has nan 
                            
                            if gt_phasor is not None and DEBUG:
                                # now plot unwrapped phase image
                                plt.figure(figsize=(12, 5))
                                plt.subplot(1, 2, 1)
                                plt.plot(unrap_tesnor(phi[start_i]).cpu(), label='Rec Row line at start_i')
                                plt.title('Sample Row Line from Recovered Phase')
                                # make gt dash line
                                plt.plot(unrap_tesnor(gt_phase[start_i]), label='GT Row line at start_i', linestyle='--')
                                plt.title('GT Sample Row Line from Recovered Phase')
                                plt.legend()
                                plt.subplot(1, 2, 2)
                                plt.plot(unrap_tesnor(phi[:, start_j]).cpu(), label='Rec Column line at start_j')
                                plt.title('Sample Column Line from Recovered Phase')
                                plt.plot(unrap_tesnor(gt_phase[:, start_j]), label='GT Column line at start_j', linestyle='--')
                                plt.title('GT Sample Column Line from Recovered Phase')
                                plt.tight_layout()
                                plt.legend()
                                plt.savefig(f'{results_dir}/{exp_name}/phase_profile_row_col_unwrapped_{shifts_str}.png', dpi=150)
                                plt.close()

                            ls_phi = metadata['phi_least_square']

                            if DEBUG:
                                plt.imshow(phi.cpu(), cmap='twilight')
                                plt.colorbar()
                                plt.tight_layout()
                                plt.savefig(f'{results_dir}/{exp_name}/recovered_phase_{shifts_str}.png', dpi=300)
                                plt.close()

                                plt.imshow(ls_phi.cpu(), cmap='twilight')
                                plt.colorbar()
                                plt.tight_layout()
                                plt.savefig(f'{results_dir}/{exp_name}/recovered_phase_ls_{shifts_str}.png', dpi=300)
                                plt.close()        

                            if USE_WANDB:
                                current_step = wandb.run.step
                                log_key = f"tau={noise_param['tau']}_std={noise_param['noise_std']}"
                                wandb.log({
                                    "shift_pair": shifts_str,
                                    f"{log_key}/mae": phase_error["mae"],
                                    f"{log_key}/mse": phase_error["mse"],
                                    f"{log_key}/mae_std": phase_error["mae_std"],
                                    f"{log_key}/circ_mae": phase_error["circ_mae"],
                                    f"{log_key}/phasor_mae" : phase_error["phasor_error"],
                                    f"{log_key}/phasor_mae_center_aligned" : phase_error["phasor_error_ls"],
                                    f"phase_image/{shifts_str}": wandb.Image(plt.gcf()),
                                })
                                phase_errors_table.add_data(
                                    current_step,
                                    shifts_str,           # Your desired tooltip text
                                    phase_error["mae"],
                                    phase_error["mse"],
                                    phase_error["circ_mae"],
                                    log_key
                                )

                                table = wandb.Table(columns=["hop", "avg_error"])
                                for hop, avg_err in zip(hops_sorted, avg_errors_all_shft):
                                    table.add_data(int(hop), float(avg_err))

                                wandb.log({"global/error_vs_hop": table})
                                # 

                            errors["shift_pairs"].append(shifts_str)
                            if noise_param is not None:
                                errors["noise_params"].append(f"tau={noise_param['tau']:.1f}_std={noise_param['noise_std']:.1f}")
                                errors["mae"].append(phase_error["mae"])
                                errors["mae_std"].append(phase_error["mae_std"])
                                errors["mse"].append(phase_error["mse"])
                                errors["circ_mae"].append(phase_error["circ_mae"])    
                                errors["phasor_error"].append(phase_error["phasor_error"])    
                                errors["phasor_error_ls"].append(phase_error["phasor_error_ls"])    
                                errors["avg_hops"].append(hops.mean())    
                            
                            
                            U0_recovered = np.exp(1j * phi.detach().cpu().numpy()) * amplitude

                            U0_final = U0_recovered.astype(np.complex64)
                            np.savez(f"{results_dir}/{exp_name}/U0_recovered_{shifts_str}.npz", U0_recovered=U0_final)

                            U0_recovered_with_ls = np.exp(1j * ls_phi.detach().cpu().numpy()) * amplitude
                            U0_final_ls = U0_recovered_with_ls.astype(np.complex64)
                            np.savez(f"{results_dir}/{exp_name}/U0_recovered_LS_{shifts_str}.npz", U0_recovered=U0_final_ls)

                            if DEBUG:
                                # plot uo amplitude and phase
                                plt.figure(figsize=(12, 5))
                                plt.subplot(1, 2, 2)
                                plt.imshow(np.angle(U0_final), cmap='twilight')
                                plt.colorbar(label='Phase (radians)')
                                plt.title('Recovered Phase of U0')
                                plt.subplot(1, 2, 1)
                                plt.imshow(np.abs(U0_final), cmap='gray')
                                plt.colorbar(label='Amplitude')
                                plt.title('Recovered Amplitude of U0')
                                plt.tight_layout()
                                plt.savefig(f'{results_dir}/{exp_name}/U0_recovered_amplitude_phase_{shifts_str}.png', dpi=150)
                                plt.close()

                            # print current error 
                            if gt_phasor is not None:
                                print(f"Phase recovery for shifts {shift_list} (noise {noise_str}): phasor_error={phase_error['phasor_error']:.4f}, phasor_error_ls={phase_error['phasor_error_ls']:.4f}")
                    
                        if USE_WANDB:
                            wandb.log({"phase_errors_data": phase_errors_table})
                        
                        if gt_phasor is not None:
                            df = pd.DataFrame(errors)
                            df.to_csv(f'{results_dir}/{base_exp_name}_{noise_str}_error_analysis.csv', index=False)
