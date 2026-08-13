import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from psi_least_sqaures import solve_2d_fourier_torch

AVG_MOD = int(os.getenv('AVG_MOD', '0'))
SAVE_RESULTS = True


def integrate_coprime_bfs_v2(phase_diff_h_list, phase_diff_v_list, shift_list,
                             amp_h_list, amp_v_list, device="cuda", MAX_Iter=50000, 
                             start_pos=(0, 0), gt_phasor=None, do_avg = True, 
                             results_dir="results", exp_name="exp"):
    shifts_str = '_'.join([str(s) for s in shift_list])
    H, W = phase_diff_h_list[0].shape
    
    phasor_sum = torch.zeros((H, W), dtype=torch.complex64, device=device)
    count = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    row_idx, col_idx = start_pos
    phasor_sum[row_idx, col_idx] = 1 #gt_phasor[row_idx, col_idx]
    count[row_idx, col_idx] = 1.0

    # minimum-hop tracker
    # -------------------------------
    hops = torch.full((H, W), float('inf'), device=device)
    hops[row_idx, col_idx] = 0
    
    from collections import deque
    queue = deque()

    # queue entry now includes hop count
    queue.append((row_idx, col_idx, 0, None, None, None, None))
    
    iteration = 0
    pbar = tqdm(total=H*W, desc="BFS Integration", disable=False)
    
    while queue and iteration < MAX_Iter:
        iteration += 1
        current_size = len(queue)
        
        for _ in range(current_size):
            i, j, curr_hops, parent_i, parent_j, parent_shift_idx, parent_dir = queue.popleft()
            
            phi_curr = phasor_sum[i, j] / count[i, j]

            
            # Try all shifts and all directions
            for k, s in enumerate(shift_list):
                diff_h = phase_diff_h_list[k]
                diff_v = phase_diff_v_list[k]

                amp_h = amp_h_list[k]
                amp_v = amp_v_list[k]
                
                # ---------- DOWN ----------
                if i + s < H:
                    if True:
                    # if not (parent_dir == 'up' and parent_shift_idx == k and parent_i == i + s):
                        # phi_new = phi_curr * torch.conj(diff_v[i, j])
                        # phi_new = phi_curr * (diff_v[i, j]) 
                        phi_new = phi_curr * (diff_v[i+s, j]) 

                        amp_new = 1 # amp_v[i+s, j]
                        phi_new *= amp_new

                        ni, nj = i + s, j
                        new_hops = curr_hops + 1

                        # HOP-TRACKING UPDATE
                        if new_hops < hops[ni, nj]:
                            hops[ni, nj] = new_hops
                            phasor_sum[ni, nj] = phi_new
                            count[ni, nj] = amp_new
                            queue.append((ni, nj, new_hops, i, j, k, 'down'))
                        elif do_avg and new_hops == hops[ni, nj]:
                            phasor_sum[ni, nj] += phi_new
                            count[ni, nj] += amp_new
                        

                # ---------- UP ----------
                if i >= s:
                    if True:
                        phi_new = phi_curr * torch.conj(diff_v[i, j])  

                        amp_new = 1 # amp_v[i, j]
                        phi_new *= amp_new

                        ni, nj = i-s, j
                        new_hops = curr_hops + 1

                        # HOP-TRACKING UPDATE
                        if new_hops < hops[ni, nj]:
                            hops[ni, nj] = new_hops
                            phasor_sum[ni, nj] = phi_new
                            count[ni, nj] = amp_new
                            queue.append((ni, nj, new_hops, i, j, k, 'up'))
                        elif do_avg and new_hops == hops[ni, nj]:
                            phasor_sum[ni, nj] += phi_new
                            count[ni, nj] += amp_new
                        
                # ---------- RIGHT ----------
                if j + s < W:
                    if True:
                        phi_new = phi_curr * torch.conj(diff_h[i, j]) # upto Feb8

                        ni, nj = i, j + s
                        new_hops = curr_hops + 1

                        amp_new = 1 # amp_h[i, j]
                        phi_new *= amp_new

                        # HOP-TRACKING UPDATE
                        if new_hops < hops[ni, nj]:
                            hops[ni, nj] = new_hops
                            phasor_sum[ni, nj] = phi_new
                            count[ni, nj] = amp_new
                            queue.append((ni, nj, new_hops, i, j, k, 'right'))
                            # print("Right move Set loc:", ni, nj, phasor_sum[ni, nj].item(), " gt: ", gt_phasor[ni, nj].item())
                        # elif do_avg and hops[ni, nj] > 0: # new_hops == hops[ni, nj]
                        elif do_avg and new_hops == hops[ni, nj]:
                            phasor_sum[ni, nj] += phi_new
                            count[ni, nj] += amp_new
                            # print("Right move Avg loc:", ni, nj, " Hops:", new_hops, " Queue size:", len(queue))
                            # exit()

                # ---------- LEFT ----------
                if j >= s:
                    if True:
                        phi_new = phi_curr * (diff_h[i, j-s])

                        ni, nj = i, j - s
                        new_hops = curr_hops + 1

                        amp_new = 1 # amp_h[i, j-s]
                        phi_new *= amp_new

                        # HOP-TRACKING UPDATE
                        if new_hops < hops[ni, nj]:
                            hops[ni, nj] = new_hops
                            phasor_sum[ni, nj] = phi_new
                            count[ni, nj] = amp_new
                            queue.append((ni, nj, new_hops, i, j, k, 'left'))
                        
                        elif do_avg and new_hops == hops[ni, nj]:
                            phasor_sum[ni, nj] += phi_new
                            count[ni, nj] += amp_new
                            # print("Left move: Avg loc:", ni, nj, " Hops:", new_hops, " Queue size:", len(queue))
                            # exit()
                
        total_filled = (count > 0).sum().item()
        pbar.n = total_filled
        pbar.set_postfix({'filled': f'{100*total_filled/(H*W):.1f}%', 'queue': len(queue)})
        pbar.refresh()
        
        if len(queue) == 0:
            break
    
    pbar.close()
    
    # Get final phase by averaging
    phasor_est = phasor_sum / (count + 1e-8)
    # phi = torch.angle(phasor_est)
    # phasor_est[count == 0] = 0# float('nan')

    phi = torch.angle(phasor_est)
    
    total_filled = (count > 0).sum().item()
    print(f"\nTotal filled: {total_filled}/{H*W} pixels ({100*total_filled/(H*W):.1f}%)")
    print(f"Average paths per pixel: {(count.sum() / max(total_filled, 1)).item():.2f}")
    print(f"Average hops per pixel: {(hops[hops < float('inf')].float().mean()).item():.2f}")
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(phi.cpu().numpy(), cmap='twilight')
    plt.colorbar(label='Phase (radians)')
    plt.title('Integrated Phase')
    
    plt.subplot(1, 3, 2)
    plt.imshow(count.cpu().numpy(), cmap='viridis')
    plt.colorbar(label='Path count')
    plt.title('Number of Paths per Pixel')
    
    err = phasor_est.cpu()*torch.conj(gt_phasor)
    global_err = torch.mean(err)
    err = torch.angle(err * torch.conj(global_err))
    # err /= err.max()
    plt.subplot(1, 3, 3)
    plt.imshow(err, cmap='twilight')
    plt.title('Error abs ')
    plt.colorbar(label='Error')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{exp_name}/bfs_integration_{shifts_str}.png', dpi=150)
    plt.close()

    
    delta_measurements = []
    two_pi = 2 * np.pi
    for k, s in enumerate(shift_list):
        delta_theta_x = phi[:, s:] - phi[:, :-s]  # (H, W-s)
        delta_s_x     = -torch.angle(phase_diff_h_list[k][:, :-s])  # SOURCE slice
        ki_x          = torch.round((delta_theta_x - delta_s_x) / two_pi)
        delta_s_hat_x = delta_s_x + two_pi * ki_x

        delta_theta_y = phi[s:, :] - phi[:-s, :]  # (H-s, W)
        delta_s_y     = torch.angle(phase_diff_v_list[k][s:, :])   # DEST slice
        ki_y          = torch.round((delta_theta_y - delta_s_y) / two_pi)
        delta_s_hat_y = delta_s_y + two_pi * ki_y

        delta_measurements.append((delta_s_hat_x, delta_s_hat_y))
    
    
    # Solve with Fourier least squares
    phi_least_square = solve_2d_fourier_torch(delta_measurements, shift_list, (H, W), device=device)

    return phi, {'hops': hops, 'path_counts': count, 'phi_least_square': phi_least_square}


def integrate_coprime_bfs_v3(phase_diff_h_list, phase_diff_v_list, shift_list, amp_h_list, amp_v_list, device="cuda", MAX_Iter=100000, start_pos=(0, 0), gt_phasor=None, do_avg=True, results_dir="results", exp_name="exp"):
    shifts_str = '_'.join([str(s) for s in shift_list])
    H, W = phase_diff_h_list[0].shape
    
    # Initialization
    phasor_sum = torch.zeros((H, W), dtype=torch.complex64, device=device)
    count = torch.zeros((H, W), dtype=torch.float32, device=device)
    hops = torch.full((H, W), float('inf'), device=device)
    
    row_idx, col_idx = start_pos
    phasor_sum[row_idx, col_idx] = 1.0
    count[row_idx, col_idx] = 1.0
    hops[row_idx, col_idx] = 0

    pbar = tqdm(total=H*W, desc="Vectorized Wavefront Integration", disable=True)

    mean_amp = torch.stack(amp_h_list + amp_v_list).mean(dim=0)
    mean_amp = torch.ones_like(mean_amp)  
    mask_valid = mean_amp > (0.0 * mean_amp.max()) #  0.2
    
    for iteration in range(MAX_Iter):
        # Identify the frontier (pixels with valid data)
        mask_active = (count > 0)
        phi_curr = phasor_sum / (count + 1e-8)
        
        # Temporary buffers for this wave's updates
        wave_phasor = torch.zeros_like(phasor_sum)
        wave_count = torch.zeros_like(count)
        wave_hops = torch.full((H, W), float('inf'), device=device)

        for k, s in enumerate(shift_list):
            diff_h = phase_diff_h_list[k]
            diff_v = phase_diff_v_list[k]
            
            # Move DOWN
            mask = mask_active[:-s, :] # & mask_valid[s:, :]
            if mask.any():
                phi_new = phi_curr[:-s, :] * diff_v[s:, :]
                wave_phasor[s:, :][mask] += phi_new[mask]
                wave_count[s:, :][mask] += 1.0
                wave_hops[s:, :][mask] = iteration + 1

            # Move UP
            mask = mask_active[s:, :] # & mask_valid[:-s, :]
            if mask.any():
                phi_new = phi_curr[s:, :] * torch.conj(diff_v[s:, :])
                wave_phasor[:-s, :][mask] += phi_new[mask]
                wave_count[:-s, :][mask] += 1.0
                wave_hops[:-s, :][mask] = iteration + 1

            # Move RIGHT
            mask = mask_active[:, :-s] # & mask_valid[:, s:]
            if mask.any():
                # phi_new = phi_curr[:, :-s] * torch.conj(diff_h[:, :-s])
                phi_new = phi_curr[:, :-s] * torch.conj(diff_h[:, :-s])
                wave_phasor[:, s:][mask] += phi_new[mask]
                wave_count[:, s:][mask] += 1.0
                wave_hops[:, s:][mask] = iteration + 1

            # Move LEFT
            mask = mask_active[:, s:] # & mask_valid[:, :-s]
            if mask.any():
                phi_new = phi_curr[:, s:] * diff_h[:, :-s]
                wave_phasor[:, :-s][mask] += phi_new[mask]
                wave_count[:, :-s][mask] += 1.0
                wave_hops[:, :-s][mask] = iteration + 1

        # Apply Updates
        new_mask = (wave_count > 0)
        # Update unvisited pixels (Minimum Hop)
        first_visit = (hops == float('inf')) & new_mask
        phasor_sum[first_visit] = wave_phasor[first_visit]
        count[first_visit] = wave_count[first_visit]
        hops[first_visit] = wave_hops[first_visit]

        # Average with existing pixels if do_avg is True and same hop level
        if do_avg:
            if AVG_MOD == 1:
                avg_cond = (hops != float('inf')) & (~first_visit) # Any hop 
            elif AVG_MOD == 2:
                avg_cond = (torch.abs(wave_hops - hops) <= 5) & (~first_visit) # Treshold
            else:
                avg_cond = (wave_hops == hops) & (~first_visit) # Same hop
            
            
            phasor_sum[avg_cond] += wave_phasor[avg_cond]
            count[avg_cond] += wave_count[avg_cond]

        total_filled = (count > 0).sum().item()
        pbar.n = total_filled
        pbar.refresh()
        
        if total_filled == H * W:
            break

    pbar.close()
    
    # Final phase extraction
    phasor_est = phasor_sum / (count + 1e-8)
    phi = torch.angle(phasor_est)
    phi[count == 0] = 0

    delta_measurements = []
    for k, s in enumerate(shift_list):
        # HORIZONTAL - Match BFS RIGHT propagation
        delta_theta_x = phi[:, s:] - phi[:, :-s]  # CHANGED: was [:, :-s] - [:, s:]
        delta_s_x = -torch.angle((phase_diff_h_list[k][:, :-s]))  # CHANGED: added conj()
        ki_x = torch.round((delta_theta_x - delta_s_x) / (2 * torch.pi))
        delta_s_hat_x = delta_s_x + 2 * torch.pi * ki_x
        
        # VERTICAL - This was already correct
        delta_theta_y = phi[s:, :] - phi[:-s, :]
        delta_s_y = torch.angle(phase_diff_v_list[k][s:, :])
        ki_y = torch.round((delta_theta_y - delta_s_y) / (2 * torch.pi))
        delta_s_hat_y = delta_s_y + 2 * torch.pi * ki_y
        
        delta_measurements.append((delta_s_hat_x, delta_s_hat_y))

    phi_least_square = solve_2d_fourier_torch(delta_measurements, shift_list, (H, W), device=device)

    if SAVE_RESULTS:    
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(phi.cpu().numpy(), cmap='twilight')
        plt.colorbar(label='Phase (radians)'); plt.title('Integrated Phase')
        
        plt.subplot(1, 3, 2)
        plt.imshow(count.cpu().numpy(), cmap='viridis')
        plt.colorbar(label='Path count'); plt.title('Number of Paths per Pixel')
        
        plt.subplot(1, 3, 3)
        plt.hist(hops[hops < float('inf')].cpu().numpy().flatten(), bins=30, color='blue', alpha=0.7)
        plt.title('Hops Histogram')
         
        plt.tight_layout()
        plt.savefig(f'{results_dir}/{exp_name}/bfs_integration_{shifts_str}.png', dpi=150)
        plt.close()
        
    return phi, {'hops': hops, 'path_counts': count, 'phi_least_square': phi_least_square}
    