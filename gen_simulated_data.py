import argparse
import zlib

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import cv2
import imageio.v3 as iio

from phase_map_gen import create_slm_phase_map, generate_phase_pattern
from propagator import propagate_ift_torch, propagate_ft_torch
from skimage import data

from noise_models import PGNoiseModel, generate_noise_sweep

import os 
from PIL import Image, ImageDraw, ImageFont
from data.div_loader import get_dataloader 


def get_PSI_measurement_fourier(U0, sign, jj, ii, slm_Nx, slm_Ny, delta_x, wavelength, focal_length):
    """
    Generates the intensity measurement I4 for a given phase and spatial shift.
    
    Args:
        U0: Input complex field (torch.Tensor)
        sign: Direction of shift (1 for Y, -1 for X)
        jj: Pixel shift value
        ii: Global phase shift multiplier (0, 0.5, 1, 1.5)
    """
    pad = 128
    U0_padded = torch.zeros((U0.shape[2] + 2*pad, U0.shape[3] + 2*pad), dtype=torch.complex64).to(U0.device)
    U0_padded[pad:-pad, pad:-pad] = U0
    U0 = U0_padded
    my_global_phase = np.pi * ii
    if sign == 1:
        my_shift_x, my_shift_y = 0, -jj
    else:
        my_shift_x, my_shift_y = jj, 0

    phi_on_fft = create_slm_phase_map(
        global_phase_shift=my_global_phase,
        shift_pixels_x=my_shift_x ,
        shift_pixels_y=my_shift_y ,
        Ny=slm_Ny+ 2 * pad, Nx=slm_Nx+ 2 * pad
    )
    phi_on_fft = torch.from_numpy(np.mod(phi_on_fft, 2*np.pi)).to(U0.device)

    U2 = propagate_ft_torch(U0, delta_x, wavelength, focal_length)
    U2_after_slm = U2 + U2 * torch.exp(1j * phi_on_fft)

    U4 = propagate_ift_torch(U2_after_slm, delta_x, wavelength, focal_length)
    U4 = U4[pad:-pad, pad:-pad]  # Crop back to original size
    I4 = torch.abs(U4)**2
    print("I4 shape: ", I4.shape)
    return I4

def get_PSI_measurement_simple(U0, sign, jj, ii):
    """
    Non-propagation version: Matches the Fourier-padded version by using 
    zero-fill shifting instead of circular rolling.
    """
    device = U0.device
    phase_shift = torch.exp(1j * torch.tensor(np.pi * ii)).to(device)
    
    if sign == 1:
        shift_y, shift_x = -jj, 0
    else:
        shift_y, shift_x = 0, jj 
        
    # Apply linear shift with zero-padding (simulates shifting out of frame)
    pad_l = max(0, shift_x)
    pad_r = max(0, -shift_x)
    pad_t = max(0, shift_y)
    pad_b = max(0, -shift_y)

    # Shift and fill with zeros
    U0_shifted = F.pad(U0, (pad_l, pad_r, pad_t, pad_b))
    
    # Crop to maintain original dimensions [..., Ny, Nx]
    # If we padded left, we crop from the right, etc.
    if shift_x > 0: U0_shifted = U0_shifted[..., :U0.shape[-1]]
    if shift_x < 0: U0_shifted = U0_shifted[..., -U0.shape[-1]:]
    if shift_y > 0: U0_shifted = U0_shifted[..., :U0.shape[-2], :]
    if shift_y < 0: U0_shifted = U0_shifted[..., -U0.shape[-2]:, :]

    # Interfere original field with the linearly shifted (zero-filled) field
    U_interfered = U0 + U0_shifted * phase_shift
    
    return torch.abs(U_interfered)**2

def compute_snr_db(clean, noisy, eps=1e-12):
    noise = noisy - clean
    signal_power = torch.mean(clean**2)
    noise_power = torch.mean(noise**2) + eps
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def shift2d_zero(x, shift_y, shift_x):
    """
    x: (..., H, W)
    shifts > 0 move content down/right
    """
    out = torch.zeros_like(x)

    H, W = x.shape[-2], x.shape[-1]

    y_src_start = max(0, -shift_y)
    y_src_end   = H - max(0, shift_y)
    x_src_start = max(0, -shift_x)
    x_src_end   = W - max(0, shift_x)

    y_dst_start = max(0, shift_y)
    y_dst_end   = H - max(0, -shift_y)
    x_dst_start = max(0, shift_x)
    x_dst_end   = W - max(0, -shift_x)

    out[..., y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
        x[..., y_src_start:y_src_end, x_src_start:x_src_end]

    return out

def get_PSI_measurement_simple(U0, sign, jj, ii):
    """
    Non-propagation version: Shifts U0 spatially and interferes it with itself.
    
    Args:
        U0: Input complex field [B, C, H, W]
        sign: 1 for Y-shift (Down), -1 for X-shift (Left)
        jj: Pixel amount to shift
        ii: Phase shift multiplier (0, 0.5, 1, 1.5)
    """
    phase_shift = torch.exp(1j * torch.tensor(np.pi * ii)).to(U0.device)
    
    if sign == 1:
        shift_y, shift_x = jj, 0
    else:
        shift_y, shift_x = 0, -jj 
        
    # Using roll for a circular shift (common in Fourier optics simulations)
    # U0_shifted = torch.roll(U0, shifts=(shift_y, shift_x), dims=(-2, -1))
    U0_shifted = shift2d_zero(U0, shift_y, shift_x)

    U_interfered = U0 + U0_shifted * phase_shift
    
    return torch.abs(U_interfered)**2

def save_images(image_list, folder_index, sign = 0, T = -1, folder_path = "", bit_depth = 16):
    os.makedirs(folder_path, exist_ok=True)

    max_val, dtype = (65535, np.uint16) if bit_depth == 16 else (255, np.uint8)

    for j, img in enumerate(image_list):
        filename = f'image_global_{folder_index}_sign_{sign}_shift_{T}_0000.tiff'
        filepath = os.path.join(folder_path, filename)

        img = np.ascontiguousarray(img)            # ensures memory layout
        if img.max() > 1:
            print("WARNING!!!!! image max before clip is ", img.max())
            print("Exiting...")
            # exit(1)
        img = (img.clip(0,1) * max_val).astype(dtype)

        cv2.imwrite(filepath, img)

ALL_PHASE_TYPES = {
    "quad":   ("quadratic", None),   # params filled in from wavelength/focal_length/camera_pitch below
    "random": ("random",    {"correlation_length": 3.0, "phase_std": 2.5}),
    "peak":   ("peaks",     {"amplitude": 4.0, "scale": 1.0}),
}
ALL_SHIFTS = [0, 1, 11, 13, 16, 17, 21, 22, 23, 31, 33, 63]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate simulated coprime-shift PSI measurements.")
    parser.add_argument("--output-dir", default="./measurements/simulated/",
                         help="Where to write simulated measurement folders.")
    parser.add_argument("--div2k-dir", default="./data/div2k_samples",
                         help="Path to a DIV2K (or any image) directory; defaults to the bundled samples.")
    parser.add_argument("--num-images", type=int, default=-1,
                         help="Number of source images to use (-1 = all).")
    parser.add_argument("--wavelength", type=float, default=532e-9)
    parser.add_argument("--focal-length", type=float, default=100e-3)
    parser.add_argument("--camera-pitch", type=float, default=3.45e-6)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--pad-size", type=int, default=128)
    parser.add_argument("--phase-types", default="quad,random,peak",
                         help="Comma-separated subset of: quad,random,peak")
    parser.add_argument("--shifts", default=",".join(str(s) for s in ALL_SHIFTS),
                         help="Comma-separated pixel shifts to render measurements for.")
    parser.add_argument("--add-noise", action=argparse.BooleanOptionalAction, default=True,
                         help="Apply the Poisson-Gaussian noise model (default: on). Use --no-add-noise for clean/debug measurements.")
    parser.add_argument("--noise-sweep-n", type=int, default=3,
                         help="Number of noise levels to sweep via generate_noise_sweep.")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=16,
                         help="Bit depth for saved TIFF measurements (16 recommended: ~256x less quantization error than 8-bit).")
    parser.add_argument("--max-val", type=float, default=10,
                         help="Intensity normalization headroom before quantization.")
    return parser.parse_args()


MEAS_FOLDER = f"./measurements/simulated/"

if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    div_loader = get_dataloader(img_dir=args.div2k_dir, num_imgs=args.num_images,
                                 crop_size=args.crop_size, pad=args.pad_size)
    noise_vals = {}

    requested_tags = [t.strip() for t in args.phase_types.split(",")]
    shift_list_arg = [int(s) for s in args.shifts.split(",")]

    for idx, (data, f_name, pad) in enumerate(div_loader):
        image_normalized, f_name = data[0].squeeze(), f_name[0]

        f_name = f_name.split('.')[0]  # remove file extension if present

        print("============================= running on file name: ", f_name)

        RESULTS_FOLDER = f"{args.output_dir}/{f_name}/"

        camera_pitch = args.camera_pitch
        camera_Nx = args.crop_size + args.pad_size * 2
        camera_Ny = args.crop_size + args.pad_size * 2
        slm_pitch = camera_pitch  # SLM pixel pitch (m)
        slm_Nx, slm_Ny = camera_Nx, camera_Ny
        wavelength = args.wavelength
        focal_length = args.focal_length

        delta_x = delta_y = camera_pitch

        quad_params = {
            "wavelength" : wavelength,
            "focal_length": focal_length,
            "pixel_size": camera_pitch
        }
        params_by_tag = {"quad": quad_params, "random": ALL_PHASE_TYPES["random"][1], "peak": ALL_PHASE_TYPES["peak"][1]}

        exp_specs = [
            (tag, ALL_PHASE_TYPES[tag][0], params_by_tag[tag])
            for tag in requested_tags
        ]


        for tag, phase_type, params in exp_specs:
            EXP_NAME = f"Exp_SIM_noisy_{tag}_phase_map"

            # Seed deterministically per (image, phase type)
            np.random.seed(zlib.crc32(f"{f_name}_{tag}".encode()))
            Phase, phase = generate_phase_pattern(
                camera_Ny, camera_Nx, phase_type, **params
            )

            print("Running experiment: ", EXP_NAME)

            U0 = image_normalized**0.5 * torch.from_numpy(Phase)
            U0 = U0.to(torch.complex64).to(device)
            U2 = propagate_ft_torch(U0, delta_x, wavelength, focal_length)

            delta_x2 = wavelength * focal_length / (camera_Nx * delta_x)
            delta_y2 = wavelength * focal_length / (camera_Ny * delta_y)

            # No noise sweep to run if noise is off -- one clean pass, not N identical copies.
            noise_params = generate_noise_sweep(args.noise_sweep_n) if args.add_noise else [None]

            for noise_param in noise_params:
                if args.add_noise:
                    noise_model = PGNoiseModel(**noise_param)
                    noise_str = noise_model.get_params_str()
                    print("Generating measurements with noise prop. :", noise_param)
                else:
                    noise_model = None
                    noise_str = "noiseless"
                    print("Generating noiseless measurements")

                for j, jj in enumerate(shift_list_arg):
                    image_captures = {}

                    for sign in [1, -1]:

                        for i, ii in enumerate([0, 1/2 , 1, 3/2 ]):
                            my_global_phase = np.pi * ii

                            if sign == 1:
                                my_shift_x = 0   # No shift in x
                                my_shift_y = -jj
                            else:
                                my_shift_x = jj
                                my_shift_y = 0 # No shift in Y

                            phi_on_fft = create_slm_phase_map(
                                global_phase_shift=my_global_phase,
                                shift_pixels_x=my_shift_x,
                                shift_pixels_y=my_shift_y,
                                Ny = slm_Ny, Nx = slm_Nx
                            )
                            phi_on_fft = torch.from_numpy(np.mod(phi_on_fft, 2*np.pi)).to(device)

                            U2_after_slm = (1.0 * U2 + 1.0 * U2 * torch.exp(1j * phi_on_fft)) #* slm_mask # / delta_x2**2

                            U4 = propagate_ift_torch(U2_after_slm, delta_x, wavelength, focal_length)[pad:-pad, pad:-pad]

                            img = torch.abs(U4 * torch.conj(U4))

                            noisy_img = noise_model(img) if args.add_noise else img

                            if args.add_noise:
                                snr_db = compute_snr_db(img, noisy_img).item()
                                noise_std = noise_param['noise_std']
                                if noise_std in noise_vals:
                                    noise_vals[noise_std].append(snr_db)
                                else:
                                    noise_vals[noise_std] = [snr_db]


                            folder_path = f"{RESULTS_FOLDER}/{EXP_NAME}_{noise_str}"

                            image_captures[f"{i},{sign},{jj}"] = noisy_img.cpu().numpy()

                    for key, noisy_img in image_captures.items():
                        i, sign, jj = key.split(',')
                        i, sign, jj = int(i), int(sign), int(jj)
                        noisy_img = noisy_img / args.max_val
                        save_images([noisy_img], i, sign, jj, folder_path = folder_path, bit_depth = args.bit_depth)
            
            U0  = U0[pad:-pad, pad:-pad]                  
            
            ground_truth_path = f"{RESULTS_FOLDER}/{EXP_NAME}_ground_truth_amp_phase.npz"
            
            # save amplitude and phase separately. Phase comes from the clean synthetic
            np.savez(ground_truth_path, amplitude=np.absolute(U0.cpu().numpy()), phase=phase[pad:-pad, pad:-pad])