

import numpy as np
import torch


def solve_2d_fourier(delta_measurements, shifts, shape, regularization=1e-10):
    H, W = shape
    
    max_shift = max(shifts)
    H_p, W_p = H + max_shift, W + max_shift
    
    numerator = np.zeros((H_p, W_p), dtype=complex)
    denominator = np.zeros((H_p, W_p), dtype=float)
    
    for (delta_x, delta_y), s in zip(delta_measurements, shifts):
        
        if delta_x is not None:
            # Horizontal diffs
            dx_padded = np.zeros((H_p, W_p))
            dx_padded[:H, s:W] = delta_x  # 
            
            # Kernel: 
            D_kernel = np.zeros((H_p, W_p))
            D_kernel[0, 0] = 1
            D_kernel[0, s] = -1
            
            D_hat = np.fft.fft2(D_kernel)
            b_hat = np.fft.fft2(dx_padded)
            
            numerator += np.conj(D_hat) * b_hat
            denominator += np.abs(D_hat) ** 2
            
        if delta_y is not None:
            dy_padded = np.zeros((H_p, W_p))
            dy_padded[s:H, :W] = delta_y  # 
            
            # Kernel vertically
            D_kernel = np.zeros((H_p, W_p))
            D_kernel[0, 0] = 1
            D_kernel[s, 0] = -1
            
            D_hat = np.fft.fft2(D_kernel)
            b_hat = np.fft.fft2(dy_padded)
            
            numerator += np.conj(D_hat) * b_hat
            denominator += np.abs(D_hat) ** 2
    
    denominator = np.maximum(denominator, regularization)
    theta_hat = numerator / denominator
    theta_recon_full = np.fft.ifft2(theta_hat).real
    theta_recon = theta_recon_full[:H, :W]
    
    return theta_recon

def solve_2d_fourier_torch(delta_measurements, shifts, shape, device='cuda', regularization=1e-5):
    """
    PyTorch GPU version of Fourier-based phase reconstruction.
    """
    H, W = shape
    
    max_shift = max(shifts)
    H_p, W_p = H + max_shift, W + max_shift
    
    numerator = torch.zeros((H_p, W_p), dtype=torch.complex64, device=device)
    denominator = torch.zeros((H_p, W_p), dtype=torch.float32, device=device)
    
    for (delta_x, delta_y), s in zip(delta_measurements, shifts):
        
        if delta_x is not None:
            dx_padded = torch.zeros((H_p, W_p), dtype=torch.float32, device=device)
            dx_padded[:H, s:W] = delta_x
            
            # Kernel
            D_kernel = torch.zeros((H_p, W_p), dtype=torch.float32, device=device)
            D_kernel[0, 0] = 1
            D_kernel[0, s] = -1
            
            D_hat = torch.fft.fft2(D_kernel)
            b_hat = torch.fft.fft2(dx_padded)
            
            numerator += torch.conj(D_hat) * b_hat
            denominator += torch.abs(D_hat) ** 2
            
        if delta_y is not None:
            # Vertical differences
            dy_padded = torch.zeros((H_p, W_p), dtype=torch.float32, device=device)
            dy_padded[s:H, :W] = delta_y
            
            # Kernel vertically
            D_kernel = torch.zeros((H_p, W_p), dtype=torch.float32, device=device)
            D_kernel[0, 0] = 1
            D_kernel[s, 0] = -1
            
            D_hat = torch.fft.fft2(D_kernel)
            b_hat = torch.fft.fft2(dy_padded)
            
            numerator += torch.conj(D_hat) * b_hat
            denominator += torch.abs(D_hat) ** 2
    
    denominator = torch.maximum(denominator, torch.tensor(regularization, device=device))
    theta_hat = numerator / denominator
    theta_recon_full = torch.fft.ifft2(theta_hat).real
    theta_recon = theta_recon_full[:H, :W]
    
    theta_recon = theta_recon - theta_recon.mean()
    theta_recon = torch.atan2(torch.sin(theta_recon), torch.cos(theta_recon))

    return theta_recon