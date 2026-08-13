"""Summary grid: recovered phase across shift sets, vs. ground truth.

Reads the U0_recovered_*.npz files recover_phase.py already saves and composes one
comparison figure per (image, phase type, noise level): one column per shift set,
with ground truth phase as the rightmost column and the measurement SNR (dB) in
the title. Run this after recover_phase.py has produced results.

Example:
    python summarize_results.py --phase-types peak --shift-groups "1;16,17;23,31,16,17"
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

import constants
from noise_models import PGNoiseModel
from recover_phase import center_crop, get_gt_phase, parse_shift_groups
from utils import load_amplitude_no_shift, phasor_phase_error


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize recover_phase.py recovery results across shift sets.")
    parser.add_argument("--meas-root", default="./measurements/simulated/",
                         help="Same root used when running gen_simulated_data.py / recover_phase.py.")
    parser.add_argument("--results-dir", default="results",
                         help="Same --results-dir used when running recover_phase.py.")
    parser.add_argument("--phase-types", default="quad,random,peak",
                         help="Comma-separated subset of: quad,random,peak")
    parser.add_argument("--shift-groups", default=None,
                         help="Semicolon-separated shift groups, e.g. '1;16,17'. Defaults to constants.SHIFT_GROUPS.")
    parser.add_argument("--variant", choices=["bfs", "ls"], default="bfs",
                         help="Which recovered phase to show: BFS propagation (default) or least-squares refined.")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--max-val", type=float, default=10,
                         help="Must match --max-val used in gen_simulated_data.py: noise is applied to "
                              "raw intensity before it's divided by this and saved as [0,1], so SNR "
                              "needs it to undo that scaling.")
    parser.add_argument("--out-name", default="all_shifts_summary.png",
                         help="Filename to save each summary figure as, inside its experiment's results folder.")
    return parser.parse_args()


def _clean_axis(ax, keep_ylabel=False):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not keep_ylabel:
        ax.set_ylabel("")


def load_recovered_phase(results_dir, exp_name, shifts_str, variant):
    suffix = "_LS" if variant == "ls" else ""
    path = f"{results_dir}/{exp_name}/U0_recovered{suffix}_{shifts_str}.npz"
    if not os.path.exists(path):
        return None
    U0 = np.load(path)["U0_recovered"]
    return np.angle(U0)


def snr_label(noise_folder, noise_str, max_val):
    """SNR (dB) of the raw measurement image, computed analytically from the PG noise
    model (PGNoiseModel.get_snr): signal = mean measured intensity, noise = the model's
    theoretical variance at that intensity. Reported instead of raw noise_std/tau, which
    aren't meaningful on their own without the signal level they were applied to."""
    m = re.search(r"noise_std=([\d.]+)_tau=([\d.]+)", noise_str)
    if not m:
        return "noiseless"

    noise_std, tau = float(m.group(1)), float(m.group(2))
    amplitude = load_amplitude_no_shift(noise_folder)
    # gen_simulated_data.py applies noise to the raw intensity, then divides by max_val
    # before saving as [0,1] -- undo that scaling to get back to the units noise_std/tau
    # were actually calibrated against.
    mean_intensity = float(np.mean(amplitude.astype(np.float64) ** 2)) * max_val
    snr_db = PGNoiseModel(noise_std=noise_std, tau=tau).get_snr(mean_intensity)
    return f"SNR = {snr_db:.1f} dB"


def summarize_one(meas_root, results_dir, img_folder, base_exp_name, noise_folder, shift_lists, variant, crop_size, out_name, max_val):
    meas_path = os.path.join(meas_root, img_folder)
    noise_str = os.path.basename(noise_folder)[len(base_exp_name) + 1:]
    exp_name = f"{img_folder}_{base_exp_name}_{noise_str}"

    try:
        gt_phase = center_crop(get_gt_phase(base_exp_name, meas_path), crop_size)
    except Exception as e:
        print(f"Skipping {exp_name}: ground truth not found ({e})")
        return

    gt_phasor = torch.exp(1j * torch.from_numpy(gt_phase).to(torch.complex64))

    # LS refinement is unreliable on noiseless data; fall back to BFS there regardless
    # of the requested variant.
    variant = "bfs" if noise_str == "noiseless" else variant

    shift_results = []
    for shift_list in shift_lists:
        shifts_str = "_".join(str(s) for s in shift_list)
        phi = load_recovered_phase(results_dir, exp_name, shifts_str, variant)
        if phi is None:
            continue
        phasor_est = torch.exp(1j * torch.from_numpy(phi).to(torch.complex64))
        phasor_err = phasor_phase_error(phasor_est, gt_phasor)
        shift_results.append((shift_list, phi, phasor_err))

    if not shift_results:
        print(f"No saved results found for {exp_name} (run recover_phase.py first), skipping.")
        return

    n = len(shift_results)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 4.2))

    for col, (shift_list, phi, phasor_err) in enumerate(shift_results):
        axes[col].imshow(phi, cmap="twilight")
        axes[col].set_title(f"shifts={shift_list}\nphasor err={phasor_err:.3f} rad", fontsize=9, pad=8)
        _clean_axis(axes[col])

    # Ground truth goes rightmost, clearly labeled.
    axes[n].imshow(gt_phase, cmap="twilight")
    axes[n].set_title("Ground truth", fontsize=9, fontweight="bold", pad=8)
    _clean_axis(axes[n])

    axes[0].set_ylabel("Recovered phase", fontsize=10)
    _clean_axis(axes[0], keep_ylabel=True)

    display_name = f"{img_folder}_{base_exp_name}"
    snr = snr_label(noise_folder, noise_str, max_val)
    fig.suptitle(f"{display_name}  ({'least-squares' if variant == 'ls' else 'BFS'} recovery)  {snr}")
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = f"{results_dir}/{exp_name}/{out_name}"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    args = parse_args()

    shift_lists = parse_shift_groups(args.shift_groups) if args.shift_groups else constants.SHIFT_GROUPS

    img_folders = sorted([d for d in os.listdir(args.meas_root) if os.path.isdir(os.path.join(args.meas_root, d))])
    print("Found", len(img_folders), f"image folders in {args.meas_root}")

    for img_folder in img_folders:
        meas_path = os.path.join(args.meas_root, img_folder)

        for tag in args.phase_types.split(","):
            base_exp_name = f"Exp_SIM_noisy_{tag}_phase_map"

            noise_folders = sorted(
                glob.glob(os.path.join(meas_path, f"{base_exp_name}_noise_std=*_tau=*"))
                + glob.glob(os.path.join(meas_path, f"{base_exp_name}_noiseless"))
            )

            for noise_folder in noise_folders:
                summarize_one(
                    args.meas_root, args.results_dir, img_folder, base_exp_name,
                    noise_folder, shift_lists, args.variant, args.crop_size, args.out_name,
                    args.max_val,
                )
