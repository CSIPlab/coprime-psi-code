import numpy as np
import os
import cv2 
from scipy.ndimage import gaussian_filter

def generate_random_phase_map(i, H=108, W=192, seed=42, save_dir="random_phase_map"):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"phase_map_{i}.npy")

    if os.path.exists(file_path):
        # Load existing map
        phase_map = np.load(file_path)
    else:
        print("Create phase map  ", i)
        # Set seed for reproducibility
        rng = np.random.default_rng(seed + i)  # vary by index but deterministic
        phase_map = rng.uniform(0, 1, size=(H, W)).astype(np.float32) #* 2 * np.pi
        np.save(file_path, phase_map)

    return phase_map * 2 * np.pi # / 10


def interpolate_phase_map(phase_map, H=108, W=192, target_H=1080, target_W=1920):
    # Bicubic interpolation using OpenCV
    upsampled = cv2.resize(
        phase_map,
        (target_W, target_H),
        interpolation=cv2.INTER_CUBIC
    )
    # print("sampled shape ", upsampled.shape)
    return upsampled



def create_slm_phase_map(global_phase_shift, shift_pixels_x, shift_pixels_y, Ny = 1080, Nx = 1920, f_defocus=None):
    u = np.arange(Nx) - Nx//2  # Frequency indices in x-direction
    v = np.arange(Ny) - Ny//2  # Frequency indices in y-direction
    U, V = np.meshgrid(u, v)

    # Phase Ramp: phi_ramp = 2*pi * (u * f_x + v * f_y)
    # Where f_x = shift_pixels_x / N and f_y = shift_pixels_y / N
    ramp_phase = 2 * np.pi * (
        (shift_pixels_x / Nx) * U +
        (shift_pixels_y / Ny) * V
    )

    if f_defocus is not None:
        x = (U - Nx/2) #/ N
        y = (V - Ny/2) #/ N
        quad_phase = np.pi * ((x**2 + y**2)) / (f_defocus * 532e-9) * (4.5e-6)**2
        
    else:
        quad_phase = 0

    # Combine all phases
    total_phase = ramp_phase + global_phase_shift + quad_phase

    # Wrap the result to the SLM's operating range [0, 2*pi]
    # The SLM only understands phases modulo 2*pi
    slm_phase_map = np.mod(total_phase, 2 * np.pi)

    return slm_phase_map

def generate_phase_pattern(H, W, pattern_type, **kwargs):
    """
    Generate phase patterns for optical simulation testing.
    
    Parameters:
    -----------
    H, W : int
        Height and width of the phase pattern
    pattern_type : str
        'quadratic', 'random', 'peaks', 'turbulence'
    **kwargs : pattern-specific parameters
        
    Returns:
    --------
    Phase : ndarray (H, W) - Complex phasor exp(1j * phase)
    phase : ndarray (H, W) - Real-valued phase in radians
    
    References:
    -----------
    [1] Ghiglia & Pritt, "Two-Dimensional Phase Unwrapping" (1998)
    [2] Goldstein et al., "Satellite radar interferometry" (1988) 
    [3] Fried, "Optical Resolution Through a Randomly Inhomogeneous Medium" (1966)
    """
    # Coordinate grids
    y = np.arange(H) - H//2
    x = np.arange(W) - W//2
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    if pattern_type == 'quadratic':
        wavelength = kwargs.get('wavelength', 500e-9)
        focal_length = kwargs.get('focal_length', 0.1)
        pixel_size = kwargs.get('pixel_size', 1e-6)
        
        phase = (np.pi / (wavelength * focal_length)) * pixel_size**2 * (X**2 + Y**2)
        
    elif pattern_type == 'random':
        correlation_length = kwargs.get('correlation_length', 5.0)  # pixels
        phase_std = kwargs.get('phase_std', 2.0)  # radians RMS
        
        # Generate spatially correlated random phase
        random_phase = np.random.randn(H, W)
        phase = gaussian_filter(random_phase, sigma=correlation_length)
        phase = phase * phase_std / np.std(phase)  # Normalize to desired RMS
        
    elif pattern_type == 'peaks':
        scale = kwargs.get('scale', 1.0)
        X_norm = X / (W / 6) * scale
        Y_norm = Y / (H / 6) * scale
        
        phase = (3 * (1 - X_norm)**2 * np.exp(-(X_norm**2) - (Y_norm + 1)**2) 
                - 10 * (X_norm/5 - X_norm**3 - Y_norm**5) * np.exp(-X_norm**2 - Y_norm**2) 
                - 1/3 * np.exp(-(X_norm + 1)**2 - Y_norm**2))
        
        phase = phase * kwargs.get('amplitude', 3.0)
        
    elif pattern_type == 'turbulence':
        # Kolmogorov atmospheric turbulence
        r0 = kwargs.get('r0', 20.0)  # Fried parameter in pixels
        L0 = kwargs.get('L0', min(H, W) / 2)  # Outer scale
        
        # Frequency grid
        fx = np.fft.fftfreq(W)
        fy = np.fft.fftfreq(H)
        FX, FY = np.meshgrid(fx, fy)
        F = np.sqrt(FX**2 + FY**2)
        F[0, 0] = 1e-10  # Avoid division by zero
        
        # Kolmogorov power spectrum with von Karman outer scale
        PSD = 0.023 * r0**(-5/3) * (F**2 + (1/L0)**2)**(-11/6)
        
        # Generate random realization
        cn = (np.random.randn(H, W) + 1j * np.random.randn(H, W)) / np.sqrt(2)
        phase_fft = cn * np.sqrt(PSD) * W * H
        phase = np.real(np.fft.ifft2(phase_fft))
        
    else:
        raise ValueError(f"Unknown pattern: {pattern_type}. Use: quadratic, random, peaks, turbulence")
    
    phase = np.angle(np.exp(1j * phase))
    Phase = np.exp(1j * phase)
    
    return Phase, phase