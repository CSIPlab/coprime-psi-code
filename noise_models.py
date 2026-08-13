import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_noise_sweep_v2(n_points=10):
    noise_params = []
    
    base = {
        'quantum_eff': 100,
        'p_0': 0,
        'analog_gain': 1.0,
        'eta_g_1': 1e-4,
        'eta_g_2': 1e-4,
    }
    
    scale_factors = np.linspace(1, 25, n_points)
    
    for scale in scale_factors:
        params = {
            'quantum_eff': base['quantum_eff'] / np.sqrt(scale),  # Lower QE = more noise
            'p_0': base['p_0'] * scale,
            'analog_gain': base['analog_gain'] * np.sqrt(scale),
            'eta_g_1': base['eta_g_1'] * scale,
            'eta_g_2': base['eta_g_2'] * scale,
        }
        noise_params.append(params)
    
    return noise_params


def generate_noise_sweep(n_points=10):
    noise_params = []

    # tau_values = np.linspace(100, 1000, n_points)[::-1]  # 
    noise_std_values = np.linspace(0, 30, n_points)  #
    tau_values = [100] * n_points

    for tau, sigma in zip(tau_values, noise_std_values):
        params = {
            "tau": float(tau),
            "noise_std": float(sigma),
        }
        noise_params.append(params)

    return noise_params

class PGNoiseModelFoi(torch.nn.Module):
    '''
    PGNoiseModel:
       Poisson-Gaussian noise model with clipping, based on Foi et al.,
       https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf
    '''

    def __init__(self, quantum_eff=100, p_0=0, analog_gain = 1.0, eta_g_1 = 8e-4, eta_g_2= 8e-4):
        super().__init__()

        self.quantum_eff = quantum_eff
        self.p_0 = p_0 # Pedestal param 
        self.analog_gain = analog_gain
        self.eta_g_1 = eta_g_1 # 
        self.eta_g_2 = eta_g_2

    def get_params_str(self):
        return (f"quantum_eff={self.quantum_eff}, p_0={self.p_0}, "
            f"analog_gain={self.analog_gain}, eta_g_1={self.eta_g_1}, "
            f"eta_g_2={self.eta_g_2}")
     
    def forward(self, image):
        y = image.clip(0,1)
        
        a = self.analog_gain / self.quantum_eff
        b = self.analog_gain**2 * self.eta_g_1 + self.eta_g_2 \
                - self.p_0 * self.analog_gain**2 / self.quantum_eff

        assert b > 0, f"variance for gaussian noise must be positive. {b} < 0"

        # Gaussian noise
        noise_g = (b**0.5) * torch.randn_like(y)
        # Poisson noise
        noisy_y = torch.poisson(y / a) * a
        
        return (noisy_y + noise_g).clip(0, 1)

    def get_variance(self, y = None, intensity = 1.0):
        if y is None: 
            y = torch.ones(1, 1, 1, 1) * intensity
        
        y = y.clip(0,1)
        a = self.analog_gain / self.quantum_eff
        b = self.analog_gain**2 * self.eta_g_1 + self.eta_g_2 \
                - self.p_0 * self.analog_gain**2 / self.quantum_eff
        
        # print("Variance params a,b:", a, b)
        return a * y + b

class PGNoiseModel(torch.nn.Module):
    '''
        PG noise model with readout noise and photon (Poisson) noise.
    '''

    def __init__(self, noise_std=40, tau=100):
        super().__init__()
        
        self.noise_std = noise_std  # SNR in dB
        self.tau = tau  # Integration time for Poisson noise scaling

    def get_params_str(self):
        return f"noise_std={self.noise_std:.4f}_tau={self.tau:.4f}"
     
    def forward(self, image):
        y = image
        
        readout_noise = self.noise_std * torch.randn_like(y)
        
        if self.tau != float('inf'):
            y_scaled = y * self.tau
            
            noisy_y = torch.where(
                y_scaled >= 0,
                torch.poisson(y_scaled.clamp(min=0)),
                -torch.poisson((-y_scaled).clamp(min=0))
            )
            
            result = (noisy_y + readout_noise) / self.tau
        else:
            result = y + readout_noise
        
        return result

    def get_variance(self, y=None, intensity=1.0):
        if y is None:
            y = torch.ones(1, 1, 1, 1) * intensity

        readout_var = (self.noise_std** 2 / self.tau** 2) 

        if self.tau != float('inf'):
            poisson_var = y / self.tau
            return poisson_var + readout_var
        else:
            return torch.ones_like(y) * readout_var
    
    def get_snr(self, intensity = 1.0):
        noise_var = self.get_variance(intensity)#.item()
        snr = intensity / np.sqrt(noise_var)
        
        return  20 * np.log10(snr + 1e-10)
                
if __name__ == "__main__":
    
    
    noise_params = generate_noise_sweep(10)
    noise_vars = []

    model = PGNoiseModel(**noise_params[0])
    x = torch.linspace(0, 1, steps=30)
    y = model(x) # ** 0.5
    print(y)
    plt.plot(x.numpy(), y.numpy(), 'o')
    plt.xlabel("Input")
    plt.ylabel("Noisy Output")
    plt.title("PG Noise std")
    plt.grid()
    plt.savefig("zzz_pg_noise_model_test_low.png", dpi=150)
    plt.close()


    model = PGNoiseModel(**noise_params[-1])
    x = torch.linspace(0, 1, steps=30)
    y = model(x) # ** 0.5
    print(y)
    plt.plot(x.numpy(), y.numpy(), 'o')
    plt.xlabel("Input")
    plt.ylabel("Noisy Output")
    plt.title("PG Noise std")
    plt.grid()
    plt.savefig("zzz_pg_noise_model_test_high.png", dpi=150)
    plt.close()
    
    for noise_param in noise_params:
        noise_model = PGNoiseModel(**noise_param) 
        noise_var = noise_model.get_variance()
        print("noise_var", noise_var.item())
        noise_vars.append(noise_var.item())
        
    plt.plot(noise_vars)
    plt.xlabel("noise_id")
    plt.ylabel("variance")
    plt.savefig("zzz_noise_vars.png", dpi=150)
    plt.close()    
        
    # noise_model = PGNoiseModel(noise_std=0, tau=500)
    for n in noise_params:
        noise_model = PGNoiseModel(**n)
        clean = torch.ones(1, 1, 64, 64) * 0.8

        # Run 10000 times
        samples = torch.stack([noise_model(clean) for _ in range(10000)])

        # Compute SNR
        signal = 0.8
        noise_std = samples.std().item()
        snr_db = 20 * np.log10(signal / noise_std)
        print(f"Empirical SNR: {snr_db:.2f} dB")
    
        
        snr_db = noise_model.get_snr(signal)

        print("SNR calc", snr_db)
    
            