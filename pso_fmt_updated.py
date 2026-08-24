#!/usr/bin/env python

import os
import numpy as np
import time
from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin, MomentTensor
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_lune
from mtuq.grid import UnstructuredGrid
from mtuq.grid.moment_tensor import to_mt
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# PSO velocity update with inertia weight, cognitive and social parameters
w_inertia = 0.5  # Inertia weight
c1 = 0.5         # Cognitive parameter
c2 = 1.0         # Social parameter

def to_Mw(M0):
    """Convert seismic moment (M0) to moment magnitude (Mw)."""
    return (2 / 3) * (np.log10(M0) - 9.1)

def to_M0(Mw):
    """Convert moment magnitude (Mw) to seismic moment (M0)."""
    return 10 ** (1.5 * Mw + 9.1)

PARAM_NAMES = ['v', 'w', 'kappa', 'sigma', 'h', 'Mw']

def make_bounds(magnitude_min, magnitude_max):

    lb = np.array([-1.0 / 3.0,-3.0 * np.pi / 8.0, 0.0, -90.0, 0.0,magnitude_min])

    ub = np.array([1.0 / 3.0, 3.0 * np.pi / 8.0, 360.0, 90.0, 1.0, magnitude_max])

    return lb, ub

def make_sources(swarm):

    v, w, kappa, sigma, h, mw = swarm.T

    rho = to_M0(mw) * np.sqrt(2.0)

    return UnstructuredGrid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt
    )

def evaluate_swarm(
        swarm,
        data_bw,
        data_sw,
        greens_bw,
        greens_sw,
        misfit_bw,
        misfit_sw,
        origin):

    sources = make_sources(swarm)

    vals_bw = np.asarray(
        misfit_bw(
            data_bw,
            greens_bw.select(origin),
            sources
        ),
        dtype=float
    )

    vals_sw = np.asarray(
        misfit_sw(
            data_sw,
            greens_sw.select(origin),
            sources
        ),
        dtype=float
    )

    total = vals_bw + vals_sw
    total = np.asarray(total, dtype=float).reshape(-1)

    total[~np.isfinite(total)] = 1e10

    return total



def update_swarm(swarm, velocity, personal_best_position, global_best_position, lb, ub):
    """
    Update swarm positions and velocities for PSO.
    """
    r1 = np.random.uniform(0, 1, swarm.shape)
    r2 = np.random.uniform(0, 1, swarm.shape)
    
    velocity = (w_inertia * velocity + 
                c1 * r1 * (personal_best_position - swarm) + 
                c2 * r2 * (global_best_position - swarm))
    
    # Update positions and enforce bounds
    swarm = np.clip(swarm + velocity, lb, ub)
    
    return swarm, velocity

def pso_mpi_full_mt(objective_function, lb, ub, 
                    swarmsize=100, maxiter=50, 
                    stagnation_limit=10, stagnation_threshold=1e-4):
    """
    MPI-parallelized Particle Swarm Optimization for full moment tensor inversion.
    """
    start_time = time.time()
    
    # Distribute particles among processes
    local_swarmsize = swarmsize // size
    if rank < (swarmsize % size):
        local_swarmsize += 1
        
    dim = len(lb)
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Initialize random seed differently for each process. The base seed is
    # the wall-clock time (ms); each rank adds its rank index so that ranks in
    # the same run are statistically independent.
    #base_seed = int(time.time() * 1000) % 2**32
    base_seed = 12345 #change pankaj
    np.random.seed(base_seed + rank)
    if rank == 0:
        print(f"PSO base seed (rank 0): {base_seed}; per-rank seed = base + rank")
    
    # Initialize local swarm
    local_swarm = np.random.uniform(low=lb, high=ub, size=(local_swarmsize, dim))
    local_velocity = np.random.uniform(low=-np.abs(ub - lb), high= np.abs(ub - lb), 
                                     size=(local_swarmsize, dim))
    
    # Initialize personal bests
    local_personal_best_position = np.copy(local_swarm)
    local_personal_best_value = np.array([objective_function(p) for p in local_personal_best_position])
    
    # Find global best
    if local_personal_best_value.size > 0:
        local_min_idx = np.argmin(local_personal_best_value)
        local_min_val = local_personal_best_value[local_min_idx]
        local_min_pos = local_personal_best_position[local_min_idx]
    else:
        local_min_val = float('inf')
        local_min_pos = np.zeros(dim)
    
    # Gather results to find global best
    all_min_vals = comm.gather(local_min_val, root=0)
    all_min_positions = comm.gather(local_min_pos, root=0)
    
    if rank == 0:
        global_min_idx = np.argmin(all_min_vals)
        global_best_value = all_min_vals[global_min_idx]
        global_best_position = all_min_positions[global_min_idx]
    else:
        global_best_value = None
        global_best_position = None
    
    # Broadcast global best
    global_best_value = comm.bcast(global_best_value, root=0)
    global_best_position = comm.bcast(global_best_position, root=0)
    
    # Stagnation detection
    stagnation_counter = 0
    
    # Main PSO loop
    for iteration in range(maxiter):
        if rank == 0:
            print(f"PSO Iteration {iteration + 1}/{maxiter}, Best misfit: {global_best_value:.12e}")
        
        # Update local swarm
        local_swarm, local_velocity = update_swarm(
            local_swarm, local_velocity, local_personal_best_position, 
            global_best_position, lb, ub
        )
        
        # Evaluate objective function
        local_current_values = np.array([objective_function(p) for p in local_swarm])
        
        # Update personal bests
        update_indices = np.where(local_current_values < local_personal_best_value)[0]
        for idx in update_indices:
            local_personal_best_position[idx] = local_swarm[idx]
            local_personal_best_value[idx] = local_current_values[idx]
        
        # Find local best from this iteration
        if local_personal_best_value.size > 0:
            local_min_idx = np.argmin(local_personal_best_value)
            local_min_val = local_personal_best_value[local_min_idx]
            local_min_pos = local_personal_best_position[local_min_idx]
        else:
            local_min_val = float('inf')
            local_min_pos = np.zeros(dim)
        
        # Gather and find new global best
        all_min_vals = comm.gather(local_min_val, root=0)
        all_min_positions = comm.gather(local_min_pos, root=0)
        
        prev_global_best = global_best_value
        
        if rank == 0:
            if min(all_min_vals) < global_best_value:
                global_min_idx = np.argmin(all_min_vals)
                global_best_value = all_min_vals[global_min_idx]
                global_best_position = all_min_positions[global_min_idx]
        
        # Broadcast updated global best
        global_best_value = comm.bcast(global_best_value, root=0)
        global_best_position = comm.bcast(global_best_position, root=0)
        
        # Check for stagnation
        if rank == 0:
            improvement = float(prev_global_best - global_best_value)
            if improvement < stagnation_threshold:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            if stagnation_counter >= stagnation_limit:
                print(f"Convergence achieved at iteration {iteration + 1}")
                break
        
        # Broadcast stagnation decision
        stagnation_counter = comm.bcast(stagnation_counter, root=0)
        if stagnation_counter >= stagnation_limit:
            break
    
    elapsed_time = time.time() - start_time
    
    if rank == 0:
        return global_best_position, global_best_value, elapsed_time
    else:
        return np.zeros(len(lb)), 0.0, elapsed_time

def pso_full_moment_tensor_inversion(data_bw, data_sw, greens_bw, greens_sw, 
                                   misfit_bw, misfit_sw, origin, stations, 
                                   event_id, process_bw, process_sw, 
                                   magnitude_min=4.0, magnitude_max=5.5):
    """
    Perform full moment tensor inversion using MPI-parallelized PSO.
    """
    # Define bounds for PSO: [Mrr_norm, Mtt_norm, Mpp_norm, Mrt_norm, Mrp_norm, Mtp_norm, Mw]
    # Normalized moment tensor components range from -1 to 1
    lb, ub = make_bounds(magnitude_min,magnitude_max)

    def objective_function(params):

        try:
            values = evaluate_swarm(
                np.asarray(params, dtype=float).reshape(1, -1),
                data_bw,
                data_sw,
                greens_bw,
                greens_sw,
                misfit_bw,
                misfit_sw,
                origin
            )

            value = float(values[0])

            if not np.isfinite(value):
                return 1e10

            return value

        except Exception as e:
            if rank == 0:
                print(f"Error in objective_function: {e}")
            return 1e10

    if rank == 0:
        print("Starting MPI-PSO full moment tensor optimization...")
    
    # Adjust swarm size for better MPI parallelization
    swarmsize = 100
    if swarmsize % size != 0:
        swarmsize = size * (swarmsize // size + 1)
    
    # Run PSO
    best_position, best_misfit, pso_time = pso_mpi_full_mt(
        objective_function, 
        lb, 
        ub,
        swarmsize=swarmsize, 
        maxiter=50, 
        stagnation_limit=10,
        stagnation_threshold=1e-4
    )
    
    # Post-process results (only on root process)
    if rank == 0:
        # Create best-fitting moment tensor
        #best_mt = create_mt_from_normalized_components(best_position)
        best_source_grid = make_sources(
            np.asarray(best_position, dtype=float).reshape(1, -1)
        )

        best_mt = best_source_grid.get(0)
        lune_dict = best_source_grid.get_dict(0)
        
        # Extract parameters
        best_Mw = float(best_position[5])
        best_M0 = to_M0(best_Mw)
        
        # Calculate final misfit
        misfit_bw_value = misfit_bw(data_bw, greens_bw.select(origin), best_mt, level =0)
        misfit_sw_value = misfit_sw(data_sw, greens_sw.select(origin), best_mt, level =0)
        
        # Use modern NumPy scalar extraction
        if hasattr(misfit_bw_value, 'item'):
            misfit_bw_scalar = misfit_bw_value.item()
        else:
            misfit_bw_scalar = float(misfit_bw_value)
            
        if hasattr(misfit_sw_value, 'item'):
            misfit_sw_scalar = misfit_sw_value.item()
        else:
            misfit_sw_scalar = float(misfit_sw_value)
            
        final_misfit = misfit_bw_scalar + misfit_sw_scalar
        
        # Get lune parameters with corrected strike/dip/rake calculation

        
        v = float(best_position[0])
        w = float(best_position[1])
        kappa = float(best_position[2])
        sigma = float(best_position[3])
        h = float(best_position[4])
        best_Mw = float(best_position[5])

        dip = np.degrees(
            np.arccos(np.clip(h, 0.0, 1.0))
        )

        print("Best solution parameters:")
        print(f"  Mw:     {best_Mw:.2f}")
        print(f"  Strike: {kappa:.1f}°")
        print(f"  Dip:    {dip:.1f}°")
        print(f"  Rake:   {sigma:.1f}°")
        print(f"  v:      {v:.6f}")
        print(f"  w:      {w:.6f}")
        print(f"  h:      {h:.6f}")
        
        # Get moment tensor dictionary
        try:
            mt_dict = best_mt.as_dict() if hasattr(best_mt, 'as_dict') else {}
        except:
            mt_dict = {}
        
        # Create solution dictionary
        best_solution = merge_dicts(
            mt_dict,
            lune_dict,
            {'M0': float(best_M0)},
            {'Mw': float(best_Mw)},
            {'misfit_value': float(final_misfit)},
            {'computation_time': float(pso_time)},
            {'num_processes': size},
            {'optimization_method': 'PSO_full_moment_tensor'},
            origin.as_dict() if hasattr(origin, 'as_dict') else {}
        )
        
        return best_solution, best_mt, final_misfit, pso_time
    else:
        return None, None, None, pso_time

def main():
    """
    Main function to execute MPI-based full moment tensor inversion using PSO.
    """
    try:
        if rank == 0:
            overall_start_time = time.time()
            print("Starting Full Moment Tensor PSO Inversion...")
        
        # Define paths and parameters
        path_data = fullpath('/home/pankaj/Pankaj_PhD_work/ANET_PSO/Anet_example/20090407201255351/*.[zrt]')
        path_weights = fullpath('/home/pankaj/Pankaj_PhD_work/ANET_PSO/Anet_example/20090407201255351/weights.dat')
        event_id = '20090407201255351'
        model = 'ak135'
        
        # Magnitude search range
        magnitude_min = 4.5
        magnitude_max = 4.5
        initial_magnitude = (magnitude_min + magnitude_max) / 2

        # Data processing for body waves
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

        # Data processing for surface waves
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

        # Misfit functions
        misfit_bw = Misfit(
            norm='L2',
            time_shift_min=-2.0,
            time_shift_max=+2.0,
            time_shift_groups=['ZR'],
            normalize=False
        )

        misfit_sw = Misfit(
            norm='L2',
            time_shift_min=-10.0,
            time_shift_max=+10.0,
            time_shift_groups=['ZR', 'T'],
            normalize=False
        )

        # Parse station codes
        station_id_list = parse_station_codes(path_weights)

        # Define origin
        origin = Origin({
            'time': '2009-04-07T20:12:55.000000Z',
            'latitude': 61.454200744628906,
            'longitude': -149.7427978515625,
            'depth_in_m': 33033.599853515625,
        })
        
        # Source time function
        wavelet = Trapezoid(magnitude=initial_magnitude)

        # Data I/O
        if rank == 0:
            print('Reading data...')
        data = read(path_data, format='sac', event_id=event_id, 
                   station_id_list=station_id_list, tags=['units:m', 'type:velocity'])
        data.sort_by_distance()
        stations = data.get_stations()

        if rank == 0:
            print('Processing data...')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        if rank == 0:
            print('Reading Greens functions...')
        greens = download_greens_tensors(stations, origin, model)

        if rank == 0:
            print('Processing Greens functions...')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

        if rank == 0:
            print('Running PSO-based full moment tensor inversion...')

        # Run PSO-based inversion
        best_solution, best_mt, final_misfit, pso_time = pso_full_moment_tensor_inversion(
            data_bw, data_sw, greens_bw, greens_sw, 
            misfit_bw, misfit_sw, origin, stations, 
            event_id, process_bw, process_sw,
            magnitude_min=magnitude_min, 
            magnitude_max=magnitude_max
        )
        
        # Post-processing (root process only)
        if rank == 0:
            magnitude_val = float(best_solution['Mw'])
            M0_val = float(best_solution['M0'])
            
            print(f'Final misfit value: {float(final_misfit):.6f}')
            print(f'Best magnitude: Mw {magnitude_val:.2f}')
            print(f'Scalar moment: M0 {M0_val:.2e} N·m')
            print(f'PSO computation time: {pso_time:.2f} seconds')
            
            # Save results
            print('Saving results...')
            save_json(event_id + '_PSO_FullMT_solution.json', best_solution)
            
            # Generate plots
            filename_base = f"{event_id}_PSO_FullMT_Mw{magnitude_val:.2f}"
            
            print('Generating figures...')
            try:
                plot_data_greens2(filename_base + '_waveforms.png', 
                                data_bw, data_sw, greens_bw, greens_sw, 
                                process_bw, process_sw, misfit_bw, misfit_sw, 
                                stations, origin, best_mt, best_solution)
                
                plot_beachball(filename_base + '_beachball.png', 
                             best_mt, stations, origin)
                
                print('Full moment tensor PSO inversion completed successfully!')
                
            except Exception as e:
                print(f"Error during plotting: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        if rank == 0:
            print(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
