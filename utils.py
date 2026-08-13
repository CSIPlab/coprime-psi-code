import os

import numpy as np
import torch
import imageio.v3 as iio
from math import gcd
from functools import reduce
from tqdm import tqdm

def are_coprime(shift_list):
    return reduce(gcd, shift_list) == 1

def phasor_phase_error(phasor_est, phasor_gt, eps=1e-12):
    def phasor_phase_error_int(phasor_est, phasor_gt, eps=1e-12):
        err = phasor_est * torch.conj(phasor_gt)
        theta = torch.mean(err)
        
        phase_err = err * torch.conj(theta)
        return float(torch.mean(torch.angle(phase_err).abs()).item())
    
    err = phasor_phase_error_int(phasor_est, phasor_gt, eps)
    err_conj = phasor_phase_error_int(torch.conj(phasor_est), phasor_gt, eps)

    return min(err, err_conj)
    

def unrap_tesnor(phase):
    # if numpy skip 
    if type(phase) == torch.Tensor:
        device = phase.device
        phase_np = phase.cpu().numpy()
    else:
        device = "cpu"
        phase_np = phase    
    
    if len(phase_np.shape) == 1:
        phase_np = np.unwrap(phase_np, axis=0)
    else:
        phase_np = np.unwrap(phase_np, axis=0)  # Unwrap rows
        phase_np = np.unwrap(phase_np, axis=1)  # Unwrap columns
    
    phase = torch.from_numpy(phase_np).to(device)
    return phase

def max_in_center_window(img, w_h, w_w):
    H, W = img.shape

    c_i = H // 2
    c_j = W // 2

    i1 = max(0, c_i - w_h // 2)
    i2 = min(H, c_i + w_h // 2)
    j1 = max(0, c_j - w_w // 2)
    j2 = min(W, c_j + w_w // 2)

    win = img[i1:i2, j1:j2]

    flat_idx = win.argmax()
    local_i, local_j = divmod(flat_idx.item(), win.shape[1])

    # convert back to global image coordinates
    global_i = i1 + local_i
    global_j = j1 + local_j

    return global_i, global_j

def remove_global_tilt(phi, mask=None):
    """Remove best-fit plane from phase"""
    H, W = phi.shape
    valid = ~torch.isnan(phi)
    if mask is not None:
        valid = valid & mask
    
    y, x = torch.meshgrid(torch.arange(H, device=phi.device), 
                          torch.arange(W, device=phi.device), indexing='ij')
    
    x_valid = x[valid].float()
    y_valid = y[valid].float()
    phi_valid = phi[valid]
    
    A = torch.stack([x_valid, y_valid, torch.ones_like(x_valid)], dim=1)
    coeffs = torch.linalg.lstsq(A, phi_valid).solution
    
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    return phi - plane

def recover_phase_remove_carrier(I0, I1, I2, I3, shift_pixels_x, shift_pixels_y, 
                                  Nx=1920, Ny=1200):
    """
    Recover phase and remove the known carrier frequency from SLM tilt
    """
    numerator = I1 - I3
    denominator = I0 - I2
    
    phase_recovered = torch.atan2(numerator, denominator)
    
    carrier_phase = np.pi * (shift_pixels_x + shift_pixels_y)
    
    phase_corrected = (phase_recovered - carrier_phase)
    
    # phase_corrected = torch.atan2(torch.sin(phase_corrected), torch.cos(phase_corrected))
    
    amplitude = 0.5 * torch.sqrt(numerator**2 + denominator**2)
    
    return amplitude, torch.exp(1j * phase_corrected)

def simple_4step_psi(I0, I1, I2, I3):
    numerator = I3 - I1
    denominator = I0 - I2
    
    phase = torch.atan2(numerator, denominator)
    phase -= phase.min()
    amplitude = 0.5 * torch.sqrt(numerator**2 + denominator**2)
    
    return amplitude, torch.exp(1j * phase)


def remove_constant_offset(field, amplitude, device="cuda"):
    phase = torch.angle(field)

    phase_np = np.unwrap(np.unwrap(phase.cpu().numpy(), axis=0), axis=1)
    
    phase_unwrapped = torch.from_numpy(phase_np).to(device)

    # Mask high amplitude
    mask = amplitude > (0.7 * amplitude.max())

    # Remove the median inside the valid high-amplitude region
    offset = torch.median(phase_unwrapped[mask])

    phase_centered = phase_unwrapped - offset

    # Wrap back
    phase_centered = torch.atan2(torch.sin(phase_centered),
                                 torch.cos(phase_centered))

    return torch.exp(1j * phase_centered)

def compute_phase_diffs_per_shift_v2(measurements, shift_list, device="cuda", offset_H = 1.5, offset_V = 1.5):
    measurements = measurements.to(device)
    
    phase_diff_h_list = []
    phase_diff_v_list = []
    amp_h_list = []
    amp_v_list = []

    for s_idx, k in enumerate(shift_list):
        # Horizontal
        I0_h, I1_h, I2_h, I3_h = measurements[s_idx, :, 1, :, :]
        amp_h, phase_diff_h = simple_4step_psi(I0_h, I1_h, I2_h, I3_h)
        
        # AUTO-CALIBRATE: Remove median phase in bright regions
        mask_h = amp_h > 0.5 * amp_h.max()
        if mask_h.sum() > 100:  # Need enough pixels
            offset_h = torch.median(torch.angle(phase_diff_h[mask_h]))
            phase_diff_h = phase_diff_h * torch.exp(-1j * offset_h)
            print(f"Shift {k} H: Auto-removed carrier = {offset_h:.4f} rad")
        
        # Vertical
        I0_v, I1_v, I2_v, I3_v = measurements[s_idx, :, 0, :, :]
        amp_v, phase_diff_v = simple_4step_psi(I0_v, I1_v, I2_v, I3_v)
        
        # AUTO-CALIBRATE vertical
        mask_v = amp_v > 0.5 * amp_v.max()
        if mask_v.sum() > 100:
            offset_v = torch.median(torch.angle(phase_diff_v[mask_v]))
            phase_diff_v = phase_diff_v * torch.exp(-1j * offset_v)
            print(f"Shift {k} V: Auto-removed carrier = {offset_v:.4f} rad")
        
        phase_diff_h_list.append(phase_diff_h)
        phase_diff_v_list.append(phase_diff_v)
        amp_h_list.append(amp_h)
        amp_v_list.append(amp_v)

    return phase_diff_h_list, phase_diff_v_list, amp_h_list, amp_v_list

def compute_phase_diffs_per_shift(measurements, shift_list, offset_H = 1.5, offset_V = 1.5, device="cuda"):
    measurements = measurements.to(device)
    
    phase_diff_h_list = []
    phase_diff_v_list = []
    amp_h_list = []
    amp_v_list = []

    CARRIER_FREQ_H = 0 * 1.3 / np.pi 
    CARRIER_FREQ_V = 0 * 1.3 / np.pi 
    
    OFFSET_H = -1 * offset_H / np.pi 
    OFFSET_V = -1 * offset_V / np.pi 

    for s_idx, k in enumerate(shift_list):        
        # Horizontal
        I0_h, I1_h, I2_h, I3_h = measurements[s_idx, :, 1, :, :]
        # _, phase_diff_h = simple_4step_psi(I0_h, I1_h, I2_h, I3_h)
        # amp_h, phase_diff_h = recover_phase_remove_carrier(I0_h, I1_h, I2_h, I3_h, k/1.8, 0) # 
        amp_h, phase_diff_h = recover_phase_remove_carrier(I0_h, I1_h, I2_h, I3_h, k*CARRIER_FREQ_H + OFFSET_H, 0) # 
        
        # Mask low-amplitude regions
        amp_threshold = 0.1 * amp_h.max()  # 20% of peak
        mask = amp_h > amp_threshold

        # phase_diff_h[~mask] = 0  # or float('nan')

        phase_diff_h_list.append(phase_diff_h)
        amp_h_list.append(amp_h)

        # Vertical
        I0_v, I1_v, I2_v, I3_v = measurements[s_idx, :, 0, :, :]
        
        amp_v, phase_diff_v = recover_phase_remove_carrier(I0_v, I1_v, I2_v, I3_v, 0, k*CARRIER_FREQ_V + OFFSET_V)  # 

        # # Mask low-amplitude regions
        # amp_threshold = 0.1 * amp_v.max()  # 20% of peak
        # mask = amp_v > amp_threshold

        # # phase_diff_v[~mask] = 0  # or float('nan')
        
        phase_diff_v_list.append(phase_diff_v)
        amp_v_list.append(amp_v)

    return phase_diff_h_list, phase_diff_v_list, amp_h_list, amp_v_list

def load_amplitude_no_shift(folder_path):
    filename = "image_global_0_sign_1_shift_0_0000.tiff"
    filepath = os.path.join(folder_path, filename)

    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")

    img_raw = iio.imread(filepath)
    max_val = np.iinfo(img_raw.dtype).max
    img = img_raw.astype(np.float32) / max_val   # normalize to [0,1]

    return img**0.5   # shape (H, W)

def load_shifted_measurements(folder_path, shift_list):
    global_phase_idx = [0, 1, 2, 3]
    num_global_phases = len(global_phase_idx)
    
    signs = [1, -1]
    num_signs = len(signs)
    
    num_shifts = len(shift_list)
    
    data = None
    
    for s_idx, jj in enumerate(tqdm(shift_list, desc="Loading shifts", disable=True)):
        for g_idx, folder_index in enumerate(global_phase_idx):
            for sign_idx, sign in enumerate(signs):
                filename = f'image_global_{folder_index}_sign_{sign}_shift_{jj}_0000.tiff'
                filepath = os.path.join(folder_path, filename)
                
                if not os.path.exists(filepath):
                    raise ValueError(f"File not found {filepath}")
                
                img_raw = iio.imread(filepath)
                max_val = np.iinfo(img_raw.dtype).max
                img = img_raw.astype(np.float32) / max_val
                
                if data is None:
                    H, W = img.shape
                    data = np.zeros((num_shifts, num_global_phases, num_signs, H, W), 
                                   dtype=np.float32)
                
                data[s_idx, g_idx, sign_idx, :, :] = img
    
    return data

def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd, x, y

