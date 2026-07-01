#!/usr/bin/env python

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin, MomentTensor
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_dc
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid

def to_Mw(M0):
    """Convert seismic moment (M0) to moment magnitude (Mw)."""
    return (2 / 3) * (np.log10(M0) - 9.1)

def to_M0(Mw):
    """Convert moment magnitude (Mw) to seismic moment (M0)."""
    return 10 ** (1.5 * Mw + 9.1)

def normalize_sdr(strike, dip, rake):
    """
    Normalize strike, dip, and rake to standard ranges:
    - strike: 0-360 degrees
    - dip: 0-90 degrees
    - rake: -180 to 180 degrees
    If dip > 90, finds the equivalent representation with dip <= 90
    """
    strike = float(strike)
    dip = float(dip)
    rake = float(rake)
    strike = strike % 360
    if dip > 90:
        strike = (strike + 180) % 360
        dip = 180 - dip
        rake = (rake + 180) % 360
        if rake > 180:
            rake -= 360
    rake = rake % 360
    if rake > 180:
        rake -= 360
    return strike, dip, rake

def create_mt_from_strike_dip_rake(strike, dip, rake, scalar_moment=None, magnitude=None):
    """
    Create a MomentTensor object from strike, dip, and rake angles.
    Always normalize the angles first to ensure valid ranges.
    """
    strike, dip, rake = normalize_sdr(strike, dip, rake)
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)
    rake_rad = np.deg2rad(rake)
    sin_dip = np.sin(dip_rad)
    cos_dip = np.cos(dip_rad)
    sin_rake = np.sin(rake_rad)
    cos_rake = np.cos(rake_rad)
    sin_strike = np.sin(strike_rad)
    cos_strike = np.cos(strike_rad)
    sin_2dip = np.sin(2 * dip_rad)
    Mrr = sin_2dip * sin_rake
    Mtt = -sin_dip * cos_rake * np.sin(2 * strike_rad) - sin_2dip * sin_rake * sin_strike**2
    Mpp = sin_dip * cos_rake * np.sin(2 * strike_rad) - sin_2dip * sin_rake * cos_strike**2
    Mrt = -cos_dip * cos_rake * cos_strike - np.cos(2 * dip_rad) * sin_rake * sin_strike
    Mrp = -cos_dip * cos_rake * sin_strike + np.cos(2 * dip_rad) * sin_rake * cos_strike
    Mtp = sin_dip * cos_rake * np.cos(2 * strike_rad) + 0.5 * sin_2dip * sin_rake * np.sin(2 * strike_rad)
    tensor = np.array([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])
    if magnitude is not None:
        scalar_moment = to_M0(magnitude)
    if scalar_moment is not None:
        mt = MomentTensor(tensor * scalar_moment)
    else:
        mt = MomentTensor(tensor)
    return mt, strike, dip, rake

def update_swarm(swarm, velocity, personal_best_position, global_best_position, lb, ub):
    r1 = np.random.uniform(0, 1, swarm.shape)
    r2 = np.random.uniform(0, 1, swarm.shape)
    velocity = 0.4 * velocity + 1.0 * r1 * (personal_best_position - swarm) + 1.5 * r2 * (global_best_position - swarm)
    swarm = np.clip(swarm + velocity, lb, ub)
    return swarm, velocity

def visualize_swarm(swarm, personal_best_position, global_best_position, iteration, event_id, misfit_values=None, output_dir=None):
    """
    Visualize the current state of the PSO swarm in the parameter space with color-coded misfit values.
    Save each projection as a separate file rather than combining them in one figure.
    """
    if output_dir is None:
        output_dir = f"{event_id}_iterations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    strike = swarm[:, 0]
    dip = swarm[:, 1]
    rake = swarm[:, 2]
    cos_dip = np.cos(np.deg2rad(dip))
    
    pb_strike = personal_best_position[:, 0]
    pb_dip = personal_best_position[:, 1]
    pb_rake = personal_best_position[:, 2]
    pb_cos_dip = np.cos(np.deg2rad(pb_dip))
    
    gb_strike = global_best_position[0]
    gb_dip = global_best_position[1]
    gb_rake = global_best_position[2]
    gb_cos_dip = np.cos(np.deg2rad(gb_dip))
    
    if misfit_values is not None:
        cmap = plt.cm.viridis_r
        norm = Normalize(vmin=np.min(misfit_values), vmax=np.max(misfit_values))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
    # Plot 1: Strike vs cos(Dip)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    if misfit_values is not None:
        scatter1 = ax1.scatter(strike, cos_dip, marker='o', c=misfit_values, cmap=cmap, norm=norm, alpha=0.9, s=50, label='Particles')
        cbar1 = fig1.colorbar(sm, ax=ax1)
        cbar1.set_label('Misfit Value', fontsize=12)
    else:
        scatter1 = ax1.scatter(strike, cos_dip, marker='o', color='#4a86e8', alpha=0.8, label='Particles')
    
    ax1.scatter(pb_strike, pb_cos_dip, marker='*', color='#4a86e8', s=80, alpha=0.5, label='Personal Best')
    ax1.scatter(gb_strike, gb_cos_dip, marker='X', color='red', s=170, label='Global Best')
    ax1.set_xlabel('Strike (degrees)', fontsize=12)
    ax1.set_ylabel('cos(Dip)', fontsize=12)
    ax1.set_xlim(0, 400)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Strike vs cos(Dip) - Iteration {iteration}", fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    plt.savefig(f"{output_dir}/{event_id}_iteration_{iteration}_strike_cosdip.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Strike vs Rake
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    if misfit_values is not None:
        scatter2 = ax2.scatter(strike, rake, marker='o', c=misfit_values, cmap=cmap, norm=norm, alpha=0.9, s=50, label='Particles')
        cbar2 = fig2.colorbar(sm, ax=ax2)
        cbar2.set_label('Misfit Value', fontsize=12)
    else:
        scatter2 = ax2.scatter(strike, rake, marker='o', color='#4a86e8', alpha=0.8, label='Particles')
    
    ax2.scatter(pb_strike, pb_rake, marker='*', color='#4a86e8', s=80, alpha=0.5, label='Personal Best')
    ax2.scatter(gb_strike, gb_rake, marker='X', color='red', s=170, label='Global Best')
    ax2.set_xlabel('Strike (degrees)', fontsize=12)
    ax2.set_ylabel('Rake (degrees)', fontsize=12)
    ax2.set_xlim(0, 400)
    ax2.set_ylim(-180, 180)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Strike vs Rake - Iteration {iteration}", fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    plt.savefig(f"{output_dir}/{event_id}_iteration_{iteration}_strike_rake.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Plot 3: cos(Dip) vs Rake
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    if misfit_values is not None:
        scatter3 = ax3.scatter(cos_dip, rake, marker='o', c=misfit_values, cmap=cmap, norm=norm, alpha=0.9, s=50, label='Particles')
        cbar3 = fig3.colorbar(sm, ax=ax3)
        cbar3.set_label('Misfit Value', fontsize=12)
    else:
        scatter3 = ax3.scatter(cos_dip, rake, marker='o', color='#4a86e8', alpha=0.8, label='Particles')
    
    ax3.scatter(pb_cos_dip, pb_rake, marker='*', color='#4a86e8', s=80, alpha=0.5, label='Personal Best')
    ax3.scatter(gb_cos_dip, gb_rake, marker='X', color='red', s=170, label='Global Best')
    ax3.set_xlabel('cos(Dip)', fontsize=12)
    ax3.set_ylabel('Rake (degrees)', fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-180, 180)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f"cos(Dip) vs Rake - Iteration {iteration}", fontsize=14)
    ax3.legend(loc='upper right', fontsize=10)
    plt.savefig(f"{output_dir}/{event_id}_iteration_{iteration}_cosdip_rake.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Optional: Misfit distribution histogram (to provide additional information)
    if misfit_values is not None:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.hist(misfit_values, bins=20, color='#4a86e8', alpha=0.7)
        ax4.axvline(np.min(misfit_values), color='red', linestyle='dashed', linewidth=2, label=f'Best Misfit: {np.min(misfit_values):.4f}')
        ax4.set_xlabel('Misfit Value', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title(f"Misfit Distribution - Iteration {iteration}", fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        plt.savefig(f"{output_dir}/{event_id}_iteration_{iteration}_misfit_hist.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)

def save_particle_positions(swarm, iteration, event_id):
    iter_dir = f"{event_id}_iterations"
    if not os.path.exists(iter_dir):
        os.makedirs(iter_dir)
    np.save(f"{iter_dir}/particles_iter_{iteration}.npy", swarm)

def create_misfit_grid(objective_function, lb, ub, resolution=50, params_to_vary=(0, 1, 2)):
    param_ranges = [np.linspace(lb[i], ub[i], resolution) for i in range(len(lb))]
    misfit_grids = []
    projections = [(0, 1), (0, 2), (1, 2)]
    for proj_idx, (idx1, idx2) in enumerate(projections):
        grid = np.zeros((resolution, resolution))
        idx3 = 3 - idx1 - idx2
        fixed_value = (lb[idx3] + ub[idx3]) / 2
        for i, val1 in enumerate(param_ranges[idx1]):
            for j, val2 in enumerate(param_ranges[idx2]):
                params = [0, 0, 0]
                params[idx1] = val1
                params[idx2] = val2
                params[idx3] = fixed_value
                grid[j, i] = objective_function(params)
        misfit_grids.append(grid)
    return param_ranges, misfit_grids

def plot_misfit_with_particles(iter_dir, event_id, iteration, particles, lb, ub, param_ranges, misfit_grids, best_position=None):
    projections = [
        ("Strike vs Dip", 0, 1, "Strike (°)", "Dip (°)"),
        ("Strike vs Rake", 0, 2, "Strike (°)", "Rake (°)"),
        ("Dip vs Rake", 1, 2, "Dip (°)", "Rake (°)")
    ]
    for i, (title, idx1, idx2, label1, label2) in enumerate(projections):
        fig, ax = plt.subplots(figsize=(10, 8))
        range1 = param_ranges[idx1]
        range2 = param_ranges[idx2]
        X, Y = np.meshgrid(range1, range2)
        contour = ax.contour(X, Y, misfit_grids[i], levels=15, cmap='viridis_r', alpha=0.7)
        contourf = ax.contourf(X, Y, misfit_grids[i], levels=20, cmap='viridis_r', alpha=0.5)
        ax.scatter(particles[:, idx1], particles[:, idx2], c='blue', s=20, alpha=0.7, label='Particles')
        if best_position is not None:
            ax.scatter(best_position[idx1], best_position[idx2], c='red', s=100, marker='*', label='Global Best', edgecolors='black', zorder=10)
        ax.set_xlabel(label1, fontsize=16)
        ax.set_ylabel(label2, fontsize=16)
        ax.set_title(f'{title} - Iteration {iteration}', fontsize=18)
        ax.set_xlim(lb[idx1], ub[idx1])
        ax.set_ylim(lb[idx2], ub[idx2])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='upper right', fontsize=14)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Misfit Value', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(f"{iter_dir}/{event_id}_particles_proj{i}_{title.replace(' ', '_')}_iter_{iteration}.png", dpi=300)
        plt.close(fig)

def pso_custom(objective_function, lb, ub, data_bw=None, data_sw=None, greens_bw=None, greens_sw=None,
               process_bw=None, process_sw=None, misfit_bw=None, misfit_sw=None, origin=None, stations=None,
               event_id="event", save_iterations=False, swarmsize=100, maxiter=50,
               stagnation_limit=10, stagnation_threshold=1e-4, debug=False, save_particles=True):
    dim = len(lb)
    swarm = np.random.uniform(low=lb, high=ub, size=(swarmsize, dim))
    velocity = np.random.uniform(low=-np.abs(np.array(ub) - np.array(lb)), high=np.abs(np.array(ub) - np.array(lb)), size=(swarmsize, dim))
    personal_best_position = np.copy(swarm)
    personal_best_value = np.array([objective_function(p) for p in personal_best_position])
    current_values = np.copy(personal_best_value)
    global_best_index = np.argmin(personal_best_value)
    global_best_position = personal_best_position[global_best_index]
    global_best_value = personal_best_value[global_best_index]
    best_history = [global_best_value]
    stagnation_counter = 0
    iter_dir = f"{event_id}_iterations"
    if not os.path.exists(iter_dir):
        os.makedirs(iter_dir)
    # Precompute misfit grids for contour visualization
    param_ranges, misfit_grids = create_misfit_grid(objective_function, lb, ub, resolution=50)
    for i, param_range in enumerate(param_ranges):
        np.save(f"{iter_dir}/param_range_{i}.npy", param_range)
    for i, grid in enumerate(misfit_grids):
        np.save(f"{iter_dir}/misfit_grid_{i}.npy", grid)
    # Initial visualizations
    if save_particles:
        save_particle_positions(swarm, 0, event_id)
        plot_misfit_with_particles(iter_dir, event_id, 0, swarm, lb, ub, param_ranges, misfit_grids, global_best_position)
        visualize_swarm(swarm, personal_best_position, global_best_position, 0, event_id, misfit_values=current_values, output_dir=iter_dir)
    for i in range(maxiter):
        if debug:
            print(f"Iteration {i + 1}/{maxiter}, Best Value: {float(global_best_value)}")
        # Save particle positions and visualizations
        if save_particles:
            save_particle_positions(swarm, i, event_id)
            plot_misfit_with_particles(iter_dir, event_id, i, swarm, lb, ub, param_ranges, misfit_grids, global_best_position)
            visualize_swarm(swarm, personal_best_position, global_best_position, i, event_id, misfit_values=current_values, output_dir=iter_dir)
        swarm, velocity = update_swarm(swarm, velocity, personal_best_position, global_best_position, lb, ub)
        current_values = np.array([objective_function(p) for p in swarm])
        update_indices = np.where(current_values < personal_best_value)[0]
        for idx in update_indices:
            personal_best_position[idx] = swarm[idx]
            personal_best_value[idx] = current_values[idx]
        prev_global_best = global_best_value
        if np.min(personal_best_value) < global_best_value:
            global_best_index = np.argmin(personal_best_value)
            global_best_position = personal_best_position[global_best_index]
            global_best_value = personal_best_value[global_best_index]
        # Save waveform plot for this iteration if improved and save_iterations is True
        if save_iterations and data_bw is not None:
            try:
                raw_strike = float(global_best_position[0])
                raw_dip = float(global_best_position[1])
                raw_rake = float(global_best_position[2])
                strike, dip, rake = normalize_sdr(raw_strike, raw_dip, raw_rake)
                scalar_moment = to_M0(4.5)
                best_mt, _, _, _ = create_mt_from_strike_dip_rake(strike, dip, rake, scalar_moment=scalar_moment)
                misfit_bw_value = misfit_bw(data_bw, greens_bw.select(origin), best_mt, optimization_level=0)
                misfit_sw_value = misfit_sw(data_sw, greens_sw.select(origin), best_mt, optimization_level=0)
                total_plot_misfit = float(misfit_bw_value + misfit_sw_value)
                lune_dict = {
                    'rho': float(scalar_moment),
                    'v': 0.0,
                    'w': 0.0,
                    'kappa': float(strike),
                    'sigma': float(dip),
                    'h': float(np.sin(np.deg2rad(rake))),
                }
                mt_dict = best_mt.as_dict()
                iter_solution = merge_dicts(
                    mt_dict,
                    lune_dict,
                    {'M0': float(scalar_moment)},
                    {'Mw': 4.5},
                    {'fixed_Mw': 4.5},
                    {'misfit_value': float(total_plot_misfit)},
                    {'strike': float(strike)},
                    {'dip': float(dip)},
                    {'rake': float(rake)},
                    origin.as_dict() if hasattr(origin, 'as_dict') else origin
                )
                save_json(f"{iter_dir}/{event_id}_iteration_{i+1}_solution.json", iter_solution)
                figure_title = f"{event_id}_iteration_{i+1}_waveforms_strike{float(strike):.1f}_dip{float(dip):.1f}_rake{float(rake):.1f}"
                plot_data_greens2(
                    f"{iter_dir}/{figure_title}.png",
                    data_bw, data_sw, greens_bw, greens_sw,
                    process_bw, process_sw, misfit_bw, misfit_sw,
                    stations, origin, best_mt, iter_solution
                )
                beachball_title = f"{event_id}_iteration_{i+1}_beachball_strike{float(strike):.1f}_dip{float(dip):.1f}_rake{float(rake):.1f}"
                plot_beachball(
                    f"{iter_dir}/{beachball_title}.png",
                    best_mt, stations, origin
                )
                if debug:
                    print(f" Iteration {i+1}, Misfit: {float(total_plot_misfit):.4f}, Strike: {float(strike):.1f}°, Dip: {float(dip):.1f}°, Rake: {float(rake):.1f}°")
            except Exception as e:
                print(f" Error saving iteration plot {i+1}: {e}")
        best_history.append(global_best_value)
        improvement = float(prev_global_best - global_best_value)
        if improvement < stagnation_threshold:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        if stagnation_counter >= stagnation_limit:
            if debug:
                print(f"Stopping early due to stagnation at iteration {i+1}")
            break
    # Save final visualizations
    if save_particles:
        save_particle_positions(swarm, i+1, event_id)
        plot_misfit_with_particles(iter_dir, event_id, i+1, swarm, lb, ub, param_ranges, misfit_grids, global_best_position)
        visualize_swarm(swarm, personal_best_position, global_best_position, i+1, event_id, misfit_values=current_values, output_dir=iter_dir)
    # Return normalized best position
    raw_strike = float(global_best_position[0])
    raw_dip = float(global_best_position[1])
    raw_rake = float(global_best_position[2])
    norm_strike, norm_dip, norm_rake = normalize_sdr(raw_strike, raw_dip, raw_rake)
    normalized_position = np.array([norm_strike, norm_dip, norm_rake])
    return normalized_position, global_best_value

def pso_based_inversion(data_bw, data_sw, greens_bw, greens_sw, misfit_bw, misfit_sw, origin, stations,
                       event_id, process_bw, process_sw, magnitude=4.5, save_iterations=False, save_particles=True):
    lb = [0, 0, -180]
    ub = [360, 90, 180]
    scalar_moment = to_M0(magnitude)
    def objective_function(params):
        try:
            strike = float(params[0])
            dip = float(params[1])
            rake = float(params[2])
            strike, dip, rake = normalize_sdr(strike, dip, rake)
            mt, _, _, _ = create_mt_from_strike_dip_rake(strike, dip, rake, scalar_moment=scalar_moment)
            misfit_value_bw = misfit_bw(data_bw, greens_bw.select(origin), mt, optimization_level=0)
            misfit_value_sw = misfit_sw(data_sw, greens_sw.select(origin), mt, optimization_level=0)
            total_misfit = float(misfit_value_bw + misfit_value_sw)
            return total_misfit
        except Exception as e:
            print(f"Error in objective_function: {e}")
            import traceback
            traceback.print_exc()
            return 1e10
    best_position, best_misfit = pso_custom(
        objective_function,
        lb,
        ub,
        data_bw=data_bw,
        data_sw=data_sw,
        greens_bw=greens_bw,
        greens_sw=greens_sw,
        process_bw=process_bw,
        process_sw=process_sw,
        misfit_bw=misfit_bw,
        misfit_sw=misfit_sw,
        origin=origin,
        stations=stations,
        event_id=event_id,
        save_iterations=save_iterations,
        swarmsize=100,
        maxiter=50,
        stagnation_limit=10,
        stagnation_threshold=1e-4,
        debug=True,
        save_particles=save_particles
    )
    best_strike = float(best_position[0])
    best_dip = float(best_position[1])
    best_rake = float(best_position[2])
    best_M0 = float(scalar_moment)
    best_Mw = float(magnitude)
    best_mt, best_strike, best_dip, best_rake = create_mt_from_strike_dip_rake(
        best_strike, best_dip, best_rake, scalar_moment=best_M0
    )
    misfit_bw_value = misfit_bw(data_bw, greens_bw.select(origin), best_mt, optimization_level=0)
    misfit_sw_value = misfit_sw(data_sw, greens_sw.select(origin), best_mt, optimization_level=0)
    final_plot_misfit = float(misfit_bw_value + misfit_sw_value)
    lune_dict = {
        'rho': float(best_M0),
        'v': 0.0,
        'w': 0.0,
        'kappa': float(best_strike),
        'sigma': float(best_dip),
        'h': float(np.sin(np.deg2rad(best_rake))),
    }
    mt_dict = best_mt.as_dict()
    best_solution = merge_dicts(
        mt_dict,
        lune_dict,
        {'M0': float(best_M0)},
        {'Mw': float(best_Mw)},
        {'fixed_Mw': float(magnitude)},
        {'misfit_value': float(final_plot_misfit)},
        {'strike': float(best_strike)},
        {'dip': float(best_dip)},
        {'rake': float(best_rake)},
        origin.as_dict() if hasattr(origin, 'as_dict') else origin
    )
    return best_solution, best_mt, final_plot_misfit

if __name__ == '__main__':
    try:
        path_data = fullpath('/home/anetmariap/mtuq/data/examples/20090407201255351/*.[zrt]')
        path_weights = fullpath('/home/anetmariap/mtuq/data/examples/20090407201255351/weights.dat')
        event_id = '20090407201255351'
        model = 'ak135'
        magnitude = 4.5
        save_iterations = True
        save_particles = True
        process_bw = ProcessData(
            filter_type='Bandpass',
            freq_min=0.1,
            freq_max=0.333,
            pick_type='taup',
            taup_model=model,
            window_type='body_wave',
            window_length=15.0,
            capuaf_file=path_weights,
        )
        process_sw = ProcessData(
            filter_type='Bandpass',
            freq_min=0.025,
            freq_max=0.0625,
            pick_type='taup',
            taup_model=model,
            window_type='surface_wave',
            window_length=150.0,
            capuaf_file=path_weights,
        )
        misfit_bw = Misfit(
            norm='L2',
            time_shift_min=-2.0,
            time_shift_max=+2.0,
            time_shift_groups=['ZR'],
        )
        misfit_sw = Misfit(
            norm='L2',
            time_shift_min=-10.0,
            time_shift_max=+10.0,
            time_shift_groups=['ZR', 'T'],
        )
        station_id_list = parse_station_codes(path_weights)
        origin = Origin({
            'time': '2009-04-07T20:12:55.000000Z',
            'latitude': 61.454200744628906,
            'longitude': -149.7427978515625,
            'depth_in_m': 33033.599853515625,
        })
        wavelet = Trapezoid(magnitude=magnitude)
        print('Reading data...\n')
        data = read(path_data, format='sac', event_id=event_id, station_id_list=station_id_list, tags=['units:m', 'type:velocity'])
        data.sort_by_distance()
        stations = data.get_stations()
        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)
        print('Reading Greens functions...\n')
        greens = download_greens_tensors(stations, origin, model)
        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)
        print('Running PSO-based inversion with fixed magnitude...\n')
        best_solution, best_mt, final_misfit = pso_based_inversion(
            data_bw, data_sw, greens_bw, greens_sw,
            misfit_bw, misfit_sw, origin, stations,
            event_id, process_bw, process_sw,
            magnitude=magnitude,
            save_iterations=save_iterations,
            save_particles=save_particles
        )
        strike_val = float(best_solution['strike'])
        dip_val = float(best_solution['dip'])
        rake_val = float(best_solution['rake'])
        print(f'\nFinal misfit value: {float(final_misfit)}')
        print(f'Best strike: {strike_val:.1f}°, dip: {dip_val:.1f}°, rake: {rake_val:.1f}°')
        if dip_val > 90:
            print(f"WARNING: Final dip value {dip_val:.1f}° is outside valid range (0-90°)")
            print("Re-normalizing final solution...")
            norm_strike, norm_dip, norm_rake = normalize_sdr(strike_val, dip_val, rake_val)
            print(f"Corrected solution: Strike={norm_strike:.1f}°, Dip={norm_dip:.1f}°, Rake={norm_rake:.1f}°")
            best_solution['strike'] = float(norm_strike)
            best_solution['dip'] = float(norm_dip)
            best_solution['rake'] = float(norm_rake)
            best_solution['kappa'] = float(norm_strike)
            best_solution['sigma'] = float(norm_dip)
            best_solution['h'] = float(np.sin(np.deg2rad(norm_rake)))
            best_mt, _, _, _ = create_mt_from_strike_dip_rake(norm_strike, norm_dip, norm_rake, scalar_moment=best_solution['M0'])
            strike_val = norm_strike
            dip_val = norm_dip
            rake_val = norm_rake
        print('Saving results...\n')
        save_json(event_id + '_PSO_solution.json', best_solution)
        final_filename_base = f"{event_id}_PSO_strike{strike_val:.1f}_dip{dip_val:.1f}_rake{rake_val:.1f}"
        print('Generating final figures...\n')
        try:
            plot_data_greens2(final_filename_base + '_waveforms.png', data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, misfit_bw, misfit_sw, stations, origin, best_mt, best_solution)
            plot_beachball(final_filename_base + '_beachball.png', best_mt, stations, origin)
            print('\nFinished\n')
        except Exception as e:
            print(f"Error during plotting: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

