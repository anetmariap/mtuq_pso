#!/usr/bin/env python
"""
MPI-parallelized PSO for double-couple moment tensor inversion.

The search parameters are strike, dip, rake, and moment magnitude.
The optimization uses the global-best PSO formulation of Shi and
Eberhart (1998), with inertia weight omega=0.5, cognitive coefficient
c1=0.5, and social coefficient c2=1.0.

The swarm is distributed across MPI processes, with each process
evaluating a subset of particles. Optimization is terminated when
the global best misfit shows insufficient improvement for the specified
number of consecutive iterations.

The moment tensor is constructed from strike, dip, and rake using the
analytical double-couple formulation adopted in this study.
"""

import os
import numpy as np
import time
from mtuq import read, download_greens_tensors
from mtuq.event import Origin, MomentTensor
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_dc
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

inertia = 0.5
c1 = 0.5
c2 = 1.0

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
    - rake: -90 to 90 degrees
    
    If dip > 90, finds the equivalent representation with dip <= 90
    """

    strike = float(strike)
    dip = float(dip)
    rake = float(rake)
    
    # Normalize strike to 0-360
    strike = strike % 360
    
    # If dip > 90, find the conjugate fault plane
    if dip > 90:
        # Calculate conjugate fault plane
        strike = (strike + 180) % 360
        dip = 180 - dip
        rake = (rake + 180) % 360
        if rake > 180:
            rake -= 360
    
    # Ensure rake is between -90 and 90
    rake = rake % 360
    if rake > 180:
        rake -= 360
    
    # Map rake to -90 to 90 range
    if rake > 90:
        rake = 180 - rake
        strike = (strike + 180) % 360
    elif rake < -90:
        rake = -180 - rake
        strike = (strike + 180) % 360
        
    return strike, dip, rake

def create_mt_from_strike_dip_rake(strike, dip, rake, scalar_moment=None, magnitude=None):
    """
    Create a MomentTensor object from strike, dip, and rake angles.
    Always normalize the angles first to ensure valid ranges.
    """
   
    strike, dip, rake = normalize_sdr(strike, dip, rake)
    
    # Convert angles from degrees to radians
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)
    rake_rad = np.deg2rad(rake)


    # Calculate the moment tensor components
    sin_dip = np.sin(dip_rad)
    cos_dip = np.cos(dip_rad)
    sin_rake = np.sin(rake_rad)
    cos_rake = np.cos(rake_rad)
    sin_strike = np.sin(strike_rad)
    cos_strike = np.cos(strike_rad)
    sin_2dip = np.sin(2 * dip_rad)

    # Compute normalized tensor components
    #Mrr = sin_2dip * sin_rake
    Mrr = sin_2dip * sin_rake
    Mtt = -sin_dip * cos_rake * np.sin(2 * strike_rad) - sin_2dip * sin_rake * sin_strike**2
    Mpp = sin_dip * cos_rake * np.sin(2 * strike_rad) - sin_2dip * sin_rake * cos_strike**2
    Mrt = -cos_dip * cos_rake * cos_strike - np.cos(2 * dip_rad) * sin_rake * sin_strike
    Mrp = -cos_dip * cos_rake * sin_strike + np.cos(2 * dip_rad) * sin_rake * cos_strike
    Mtp = sin_dip * cos_rake * np.cos(2 * strike_rad) + 0.5 * sin_2dip * sin_rake * np.sin(2 * strike_rad)

    Mrp = -Mrp 
    Mtp = -Mtp 


    # Create tensor in MTUQ order
    tensor = np.array([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])

    # Calculate scalar moment for proper scaling
    if magnitude is not None:
        scalar_moment = to_M0(magnitude)

    if scalar_moment is not None:
        # Scale the tensor properly by scalar moment
        mt = MomentTensor(tensor * scalar_moment)
    else:
        mt = MomentTensor(tensor)

    return mt, strike, dip, rake  # Return normalized angles along with moment tensor

def update_swarm(swarm, velocity, personal_best_position, global_best_position, lb, ub):
    """
    Update swarm positions and velocities for PSO.
    """
    r1 = np.random.uniform(0, 1, swarm.shape)
    r2 = np.random.uniform(0, 1, swarm.shape)
    velocity = inertia * velocity + c1 * r1 * (personal_best_position - swarm) + c2 * r2 * (global_best_position - swarm)
    swarm = np.clip(swarm + velocity, lb, ub)
    return swarm, velocity

def pso_mpi(objective_function, lb, ub, data_bw=None, data_sw=None, greens_bw=None, greens_sw=None, 
            process_bw=None, process_sw=None, misfit_bw=None, misfit_sw=None, origin=None, stations=None,
            event_id="event", swarmsize=100, maxiter=50, 
            stagnation_limit=10, stagnation_threshold=1e-4, debug=False):
    """
    MPI-parallelized Particle Swarm Optimization (PSO) implementation with stagnation detection.
    Each MPI process evaluates a subset of particles.
    """
    # Start timing
    start_time = time.time()
    
    # Ensure swarmsize is divisible by number of processes for simplicity
    # If not perfectly divisible, some processes will have one extra particle
    local_swarmsize = swarmsize // size
    if rank < (swarmsize % size):
        local_swarmsize += 1
        
    # Set dimension and bounds
    dim = len(lb)
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Initialize random number generator with different seeds for each process.
    # The base seed is the wall-clock time (ms); each rank adds its rank index
    # so that ranks in the same run are statistically independent.
    #base_seed = int(time.time() * 1000) % 2**32 or we can use
    # Fixed seed for reproducibility; each MPI rank receives a
    # distinct deterministic seed.
    base_seed = 12345
    np.random.seed(base_seed + rank)
    if rank == 0:
        print(f"PSO base (rank 0): {base_seed}; per-rank seed = base + rank")
    
    # Create local swarm with different random positions on each process
    local_swarm = np.random.uniform(low=lb, high=ub, size=(local_swarmsize, dim))
    local_velocity = np.random.uniform(low=-np.abs(ub - lb), high=np.abs(ub - lb), size=(local_swarmsize, dim))
    
    # Initialize personal best positions and values for local particles
    local_personal_best_position = np.copy(local_swarm)
    
    # Evaluate the objective function for local particles
    local_personal_best_value = np.array([objective_function(p) for p in local_personal_best_position])
    
    # Find global best across all processes
    if local_personal_best_value.size > 0:
        local_min_idx = np.argmin(local_personal_best_value)
        local_min_val = local_personal_best_value[local_min_idx]
        local_min_pos = local_personal_best_position[local_min_idx]
    else:
        # Handle edge case where a process might not have any particles
        local_min_val = float('inf')
        local_min_pos = np.zeros(dim)
    
    # Gather results from all processes to find the global best
    all_min_vals = comm.gather(local_min_val, root=0)
    all_min_positions = comm.gather(local_min_pos, root=0)
    
    if rank == 0:
        global_min_idx = np.argmin(all_min_vals)
        global_best_value = all_min_vals[global_min_idx]
        global_best_position = all_min_positions[global_min_idx]
    else:
        global_best_value = None
        global_best_position = None
    
    # Broadcast the global best to all processes
    global_best_value = comm.bcast(global_best_value, root=0)
    global_best_position = comm.bcast(global_best_position, root=0)
    
    # For stagnation detection
    best_history = [global_best_value]
    stagnation_counter = 0
    
    # Main PSO loop
    for i in range(maxiter):
        # Update the local swarm
        local_swarm, local_velocity = update_swarm(
            local_swarm, local_velocity, local_personal_best_position, 
            global_best_position, lb, ub
        )
        
        # Evaluate objective function for updated positions
        local_current_values = np.array([objective_function(p) for p in local_swarm])
        
        # Update personal bests for local particles
        update_indices = np.where(local_current_values < local_personal_best_value)[0]
        for idx in update_indices:
            local_personal_best_position[idx] = local_swarm[idx]
            local_personal_best_value[idx] = local_current_values[idx]
        
        # Find the local best from this iteration
        if local_personal_best_value.size > 0:
            local_min_idx = np.argmin(local_personal_best_value)
            local_min_val = local_personal_best_value[local_min_idx]
            local_min_pos = local_personal_best_position[local_min_idx]
        else:
            local_min_val = float('inf')
            local_min_pos = np.zeros(dim)
        
        # Gather results from all processes
        all_min_vals = comm.gather(local_min_val, root=0)
        all_min_positions = comm.gather(local_min_pos, root=0)
        
        # Find the new global best on the root process
        prev_global_best = global_best_value
        
        if rank == 0:
            if min(all_min_vals) < global_best_value:
                global_min_idx = np.argmin(all_min_vals)
                global_best_value = all_min_vals[global_min_idx]
                global_best_position = all_min_positions[global_min_idx]
        
        # Broadcast the updated global best to all processes
        global_best_value = comm.bcast(global_best_value, root=0)
        global_best_position = comm.bcast(global_best_position, root=0)
        
        # Check for stagnation (only on root process to avoid synchronization issues)
        if rank == 0:
            best_history.append(global_best_value)
            improvement = float(prev_global_best - global_best_value)
            
            if improvement < stagnation_threshold:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            if stagnation_counter >= stagnation_limit:
                break
        
        # Broadcast stagnation decision to all processes
        stagnation_counter = comm.bcast(stagnation_counter, root=0)
        if stagnation_counter >= stagnation_limit:
            break
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    if rank == 0:
        # Return normalized position for rank 0 only
        result = global_best_position.copy()
        
        # Normalize the SDR parameters (first 3 parameters)
        raw_strike = float(global_best_position[0])
        raw_dip = float(global_best_position[1])
        raw_rake = float(global_best_position[2])
        
        # Always normalize before returning
        norm_strike, norm_dip, norm_rake = normalize_sdr(raw_strike, raw_dip, raw_rake)
        
        # Update the first 3 parameters with normalized values
        result[0] = norm_strike
        result[1] = norm_dip
        result[2] = norm_rake
        
        return result, global_best_value, elapsed_time
    else:
        # Non-root processes return placeholder values
        return np.zeros(len(lb)), 0.0, elapsed_time

def pso_based_inversion(data_bw, data_sw, greens_bw, greens_sw, misfit_bw, misfit_sw, origin, stations, 
                        event_id, process_bw, process_sw, magnitude_min, magnitude_max):
    """
    Perform moment tensor inversion using MPI-parallelized PSO with magnitude search.
    """
    # Define bounds for PSO (strike, dip, rake, magnitude)
    lb = [0, 0, -90, magnitude_min]  # Lower bounds with magnitude - changed rake to -90
    ub = [360, 90, 90, magnitude_max]  # Upper bounds with magnitude - changed rake to 90

    def objective_function(params):
        try:
            # Convert numpy array elements to Python float
            strike = float(params[0])
            dip = float(params[1])
            rake = float(params[2])
            magnitude = float(params[3])  # Extract magnitude from params
            
            # Normalize angles to standard ranges
            strike, dip, rake = normalize_sdr(strike, dip, rake)
            
            # Calculate scalar moment from magnitude
            scalar_moment = to_M0(magnitude)
            
            # Create moment tensor with current parameters
            mt, _, _, _ = create_mt_from_strike_dip_rake(strike, dip, rake, scalar_moment=scalar_moment)
            
            # Calculate misfit values for body waves and surface waves
            misfit_value_bw = misfit_bw(data_bw, greens_bw.select(origin), mt, level=0)
            misfit_value_sw = misfit_sw(data_sw, greens_sw.select(origin), mt, level=0)
            
            # Combined misfit
            total_misfit = float(misfit_value_bw + misfit_value_sw)
            return total_misfit
        except Exception as e:
            if rank == 0:
                print(f"Error in objective_function: {e}")
                import traceback
                traceback.print_exc()
            return 1e10

    # Run MPI-parallelized PSO with stagnation detection
    if rank == 0:
        print(f"Starting MPI-PSO optimization with magnitude search...")
    
    # Adjust swarm size for better parallelization - make it a multiple of MPI size
    swarmsize = 100
    if swarmsize % size != 0:
        swarmsize = size * (swarmsize // size + 1)
    
    best_position, best_misfit, pso_time = pso_mpi(
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
        swarmsize=swarmsize, 
        maxiter=50, 
        stagnation_limit=10,
        stagnation_threshold=1e-4,
        debug=False  # Set debug to False to remove runtime display
    )
    
    # Only the root process continues with post-processing
    if rank == 0:
        # Extract best-fitting parameters - already normalized by pso_mpi
        best_strike = float(best_position[0])
        best_dip = float(best_position[1])
        best_rake = float(best_position[2])
        best_Mw = float(best_position[3])  # Extract optimized magnitude
        
        # Calculate moment from optimized magnitude
        best_M0 = to_M0(best_Mw)
        
        # Create moment tensor with normalized angles and optimized magnitude
        best_mt, best_strike, best_dip, best_rake = create_mt_from_strike_dip_rake(
            best_strike, best_dip, best_rake, scalar_moment=best_M0
        )

        #print("\nPSO Moment Tensor")
        #print(best_mt.as_dict())

        # Calculate final misfit using the exact same method as in plot_data_greens2
        misfit_bw_value = misfit_bw(data_bw, greens_bw.select(origin), best_mt, level=0)
        misfit_sw_value = misfit_sw(data_sw, greens_sw.select(origin), best_mt, level=0)
        final_plot_misfit = float(misfit_bw_value + misfit_sw_value)

        # Create output dictionary with normalized angles
        lune_dict = {
            'rho': float(best_M0),
            'v': 0.0,
            'w': 0.0,
            'kappa': float(best_strike),  # Normalized strike
            #'sigma': float(best_dip),     # Normalized dip
            'sigma': float(best_rake),
            'h': float(np.cos(np.deg2rad(best_dip))),
            #'h': float(np.sin(np.deg2rad(best_rake))),  # Based on normalized rake
        }

        mt_dict = best_mt.as_dict()
        best_solution = merge_dicts(
            mt_dict,
            lune_dict,
            {'M0': float(best_M0)},
            {'Mw': float(best_Mw)},
            {'optimized_Mw': float(best_Mw)},  # Note this is now an optimized value
            {'misfit_value': float(final_plot_misfit)},
            {'strike': float(best_strike)},
            {'dip': float(best_dip)},
            {'rake': float(best_rake)},
            {'computation_time': float(pso_time)},
            {'num_processes': size},
            origin.as_dict() if hasattr(origin, 'as_dict') else origin
        )

        return best_solution, best_mt, final_plot_misfit, pso_time
    else:
        # Non-root processes return placeholder values
        return None, None, None, pso_time

def main():
    """
    Main function to execute the MPI-based seismic inversion workflow.
    """
    try:
        # Overall timing for the entire process (only track on rank 0)
        if rank == 0:
            overall_start_time = time.time()
        
        # Define paths and parameters
        path_data = fullpath('/home/pankaj/Pankaj_PhD_work/ANET_PSO/Anet_example/20090407201255351/*.[zrt]')
        path_weights = fullpath('/home/pankaj/Pankaj_PhD_work/ANET_PSO/Anet_example/20090407201255351/weights.dat')
        event_id = '20090407201255351'
        model = 'ak135'
        
        # Define search range for magnitude (instead of fixed value)
        magnitude_min = 4.5
        magnitude_max = 4.5
        
        # For initialization and wavelet creation, use a middle value
        initial_magnitude = (magnitude_min + magnitude_max) / 2

        # Define data processing for body waves
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

        # Define data processing for surface waves
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

        # Define misfit functions
        misfit_bw = Misfit(
            norm='L2',
            time_shift_min=-2.0,
            time_shift_max=+2.0,
            time_shift_groups=['ZR'],
            normalize=False,
        )

        misfit_sw = Misfit(
            norm='L2',
            time_shift_min=-10.0,
            time_shift_max=+10.0,
            time_shift_groups=['ZR', 'T'],
            normalize=False,
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

        # Define source time function using initial magnitude
        # Note: The wavelet will be rescaled based on the optimized magnitude during inversion
        wavelet = Trapezoid(magnitude=initial_magnitude)

        if rank == 0:
            print('Reading data...')
        data = read(path_data, format='sac', event_id=event_id, station_id_list=station_id_list, tags=['units:m', 'type:velocity'])
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
            print(f'Running MPI-parallelized PSO-based inversion...')

        # Run MPI-parallelized PSO-based inversion with magnitude search
        best_solution, best_mt, final_misfit, pso_time = pso_based_inversion(
            data_bw, data_sw, greens_bw, greens_sw, 
            misfit_bw, misfit_sw, origin, stations, 
            event_id, process_bw, process_sw,
            magnitude_min=magnitude_min, 
            magnitude_max=magnitude_max
        )
        
        # Only the root process continues with post-processing
        if rank == 0:
            # Extract the solution values
            strike_val = float(best_solution['strike'])
            dip_val = float(best_solution['dip'])
            rake_val = float(best_solution['rake'])
            magnitude_val = float(best_solution['Mw'])
            
            print(f'Final misfit value: {float(final_misfit)}')
            print(f'Best strike: {strike_val:.1f}°, dip: {dip_val:.1f}°, rake: {rake_val:.1f}°')
            print(f'Best magnitude: Mw {magnitude_val:.2f}')
            
            # Double check that the values are properly normalized
            if dip_val > 90 or rake_val > 90 or rake_val < -90:
                print("Re-normalizing final solution...")
                norm_strike, norm_dip, norm_rake = normalize_sdr(strike_val, dip_val, rake_val)
                
                # Update the solution before saving and plotting
                best_solution['strike'] = float(norm_strike)
                best_solution['dip'] = float(norm_dip) 
                best_solution['rake'] = float(norm_rake)
                #best_solution['kappa'] = float(norm_strike)
                best_solution['kappa'] = float(norm_strike)
                #best_solution['sigma'] = float(norm_dip) 
                #best_solution['h'] = float(np.sin(np.deg2rad(norm_rake))) 
                best_solution['sigma'] = float(norm_rake)
                best_solution['h'] = float(np.cos(np.deg2rad(norm_dip)))
                
                # Create new moment tensor with normalized values but keep the optimized magnitude
                best_mt, _, _, _ = create_mt_from_strike_dip_rake(
                    norm_strike, norm_dip, norm_rake, scalar_moment=best_solution['M0']
                )
                
                strike_val = norm_strike
                dip_val = norm_dip
                rake_val = norm_rake

            print('Saving results...')
            save_json(event_id + '_MPI_PSO_solution_with_magnitude.json', best_solution)

            # Include SDR and magnitude values in final figure filenames
            final_filename_base = f"{event_id}_MPI_PSO_strike{strike_val:.1f}_dip{dip_val:.1f}_rake{rake_val:.1f}_Mw{magnitude_val:.2f}"
            
            print('Generating final figures...')
            try:
                plot_data_greens2(final_filename_base + '_waveforms.png', data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, misfit_bw, misfit_sw, stations, origin, best_mt, best_solution)
                plot_beachball(final_filename_base + '_beachball.png', best_mt, stations, origin)
                print('Finished')
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
    print(f"Rank {rank} exited main()")
