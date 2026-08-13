import numpy as np
import torch
import torch.fft as tfft

def torch_fft2c(x: torch.Tensor) -> torch.Tensor:
    """2D centered FFT with 'ortho' norm (fftshift -> fft2 -> ifftshift)."""
    return tfft.fftshift(
        tfft.fft2(tfft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def torch_ifft2c(x: torch.Tensor) -> torch.Tensor:
    """2D centered IFFT with 'ortho' norm (fftshift -> ifft2 -> ifftshift)."""
    return tfft.fftshift(
        tfft.ifft2(tfft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
# Propagtion operators for GPU
# Prefactors below are dropped on purpose: they're global constant and invisible to intensity/phase-diff outputs.
def propagate_ft_torch(U: torch.Tensor,
                       dx: float,
                       wavelength: float,
                       f: float) -> torch.Tensor:
    k = 2 * np.pi / wavelength
    prefactor = np.exp(1j * 2 * k * f) / (1j * wavelength * f)
    return torch_fft2c(U) #* prefactor * dx**2

def propagate_ift_torch(U_fourier, dx, wavelength, f):
    k = 2 * np.pi / wavelength
    prefactor = (1j * wavelength * f) * np.exp(-1j * 2 * k * f)
    
    Ny, Nx = U_fourier.shape[-2:]
    # Physical frequency steps in the Fourier plane
    dfx = 1.0 / (Nx * dx)
    dfy = 1.0 / (Ny * dx) 
    
    return torch_ifft2c(U_fourier) #* prefactor * (dfx * dfy) * (Nx * Ny)
